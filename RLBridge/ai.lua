--- RLBridge AI controller module
--- Main AI loop that handles state monitoring, action execution, and file-based communication
--- Runs every frame and coordinates between game state observation and AI decisions

local AI = {}

local output = require("output")
local action = require("actions")
local communication = require("communication")
local utils = require("utils")

-- State management
local last_combined_hash = nil
local pending_action = nil
local rl_training_active = false
local last_key_pressed = nil
local cash_out_executed = false
local retry_count = 0

local blind_forced = false
local blind_force_timer = 0

local SHOP_DEBUG = false -- Set to true to enable shop state debug logging

local function is_pack_state(state)
    return state == G.STATES.TAROT_PACK
        or state == G.STATES.PLANET_PACK
        or state == G.STATES.SPECTRAL_PACK
        or state == G.STATES.STANDARD_PACK
        or state == G.STATES.BUFFOON_PACK
end

--- Initialize AI system
--- Sets up communication and prepares the AI for operation
--- @return nil
function AI.init()
    communication.init()

    -- Hook into Love2D keyboard events
    if love and love.keypressed then
        local original_keypressed = love.keypressed
        love.keypressed = function(key)
            -- Store the key press for our AI
            last_key_pressed = key

            -- Call original function
            if original_keypressed then
                original_keypressed(key)
            end
        end
    end
end

--- Main AI update loop (called every frame)
--- Monitors game state changes, handles communication, and executes AI actions
--- @return nil
function AI.update()
    if win_screen_active then
        win_screen_timer = win_screen_timer + (G.FPS_CAP and (1/G.FPS_CAP) or 0.016)
        if win_screen_timer > 10.0 then
            utils.log_ai("Win screen timeout — forcing resume")
            win_screen_active = false
            win_screen_timer = 0
        end
        return
    end
    if G.CONTROLLER.locking_buttons or G.CONTROLLER.interrupt_inputs or G.CONTROLLER.locks[1] then
        return 
    end
    if last_key_pressed then
        if last_key_pressed == "r" then
            if not rl_training_active then
                rl_training_active = true
                utils.log_ai("\n\nRL Training STARTED (R pressed)")
            end
        end
        -- Clear the key press
        last_key_pressed = nil
    end

    -- Don't process AI requests unless RL training is active
    if not rl_training_active then
        return
    end

    -- Get current game state
    local current_state = output.get_game_state()
    local available_actions = action.get_available_actions()
    
    if SHOP_DEBUG then
    -- -Comment out when not debugging
        if current_state.state == G.STATES.MENU or current_state.state == G.STATES.BLIND_SELECT then
            blind_forced = false
            blind_force_timer = 0
        end
        if not blind_forced and G.STATE == G.STATES.SELECTING_HAND then
            local input = require("input")
            input.force_beat_blind()
            blind_forced = true
        end
    end

    if current_state.state ~= G.STATES.ROUND_EVAL then
        cash_out_executed = false
        mr_bones_cashed_out = false
        last_combined_hash = nil
    end

    -- Don't continue if state = -1
        if current_state.state == -1 then
            return
        end

        -- Auto-skip MUST come before empty actions check
        if AI.should_auto_skip(current_state, available_actions) then
            AI.execute_auto_skip_action(current_state, available_actions)
            last_combined_hash = AI.hash_combined_state(output.get_game_state(), action.get_available_actions())
            return
        end

        -- Don't continue if there are no actions for the AI to do
        if next(available_actions) == nil then
            return
        end

    -- Create combined hash to detect meaningful changes
    local combined_hash = AI.hash_combined_state(current_state, available_actions)

    if combined_hash ~= last_combined_hash then
        -- Game state or available actions have changed
        utils.log_ai("State/Actions changed: State: " ..
            current_state.state .. " (" .. utils.get_state_name(current_state.state) .. ") | " ..
            "Actions: " .. table.concat(utils.get_action_names(available_actions), ", "))

        action.reset_state()

        if AI.should_auto_skip(current_state, available_actions) then
            AI.execute_auto_skip_action(current_state, available_actions)
            last_combined_hash = AI.hash_combined_state(output.get_game_state(), action.get_available_actions())
            return
        end

        -- Add retry_count to current state
        current_state.retry_count = retry_count
        
        -- Request action from AI (only if NOT auto-skipped)
        local ai_response = communication.request_action(current_state, available_actions)

        if ai_response then
            pending_action = ai_response
            last_combined_hash = combined_hash
        end
    end

    -- Execute pending action
    if pending_action then
        -- Validate action is still available in current state
        local current_actions = action.get_available_actions()
        local action_still_valid = false
        for _, valid_action in ipairs(current_actions) do
            if valid_action == pending_action.action then
                action_still_valid = true
                break
            end
        end
        
        if action_still_valid then
            local exec_params = pending_action.params or {}
            if pending_action.seed then
                exec_params.seed = pending_action.seed
            end
            local result = action.execute_action(pending_action.action, exec_params)
            if result.success then
                utils.log_ai("Action executed successfully: " .. pending_action.action)
                retry_count = 0
                pending_action = nil
                utils.log_ai("\n\n\n")
            else
                utils.log_ai("Action failed: " .. (result.error or "Unknown error"))
                retry_count = retry_count + 1
                utils.log_ai("Retry count: " .. retry_count)
                last_combined_hash = nil
            end
        else
            utils.log_ai("Action no longer valid (state changed), discarding: " .. pending_action.action)
            retry_count = 0  -- Reset on state change
            pending_action = nil
            utils.log_ai("\n\n\n")
        end
    end
end

--- Create combined hash of game state and actions for change detection
--- Only sends AI requests when both state and actions are ready/changed
--- @param game_state table Current game state data
--- @param available_actions table Available actions list
--- @return string Combined hash representing state + actions
function AI.hash_combined_state(game_state, available_actions)
    if not game_state or (G.STATE == G.STATES.SHOP and not G.shop) then
        return "WAITING"
    end
    -- State components
    local state_parts = {
        G.STATE or 0,
        game_state.state or 0,
        game_state.gold or 0,
        game_state.chips or 0,
        game_state.blind_chips or 0,
        (game_state.round and game_state.round.hands_left) or 0,
        (game_state.round and game_state.round.discards_left) or 0,
        (game_state.hand and game_state.hand.size) or 0,
        (game_state.hand and game_state.hand.highlighted_count) or 0,
        (game_state.shop and #game_state.shop.items) or 0,
        game_state.game_over or 0,
        #(game_state.consumables or {}),
        #(game_state.pack_cards or {}),
        game_state.pack_choices or 0,
    }

    -- Action components
    local action_ids = {}
    for _, id in ipairs(available_actions) do
        table.insert(action_ids, tostring(id))
    end
    table.sort(action_ids)

    -- Combine everything
    local combined = table.concat(state_parts, "_") .. "|" .. table.concat(action_ids, ",")
    return combined
end

--- Check if current state should be auto-skipped (not sent to AI)
--- @param current_state table Current game state
--- @param available_actions table Available actions list
--- @return boolean True if should auto-skip, false if send to AI
function AI.should_auto_skip(current_state, available_actions)
    -- Never auto-skip pack states — AI must decide which cards to select
    if is_pack_state(current_state.state) then
        return false
    end

    -- Auto-skip START_RUN in menu (action ID = 4)
    if current_state.state == G.STATES.MENU and #available_actions == 1 and available_actions[1] == 4 then
        return true
    end

    -- -- Auto-skip SELECT_BLIND in blind selection (action ID = 5)
    -- if current_state.state == G.STATES.BLIND_SELECT and #available_actions == 1 and available_actions[1] == 5 then
    --     return true
    -- end
    if current_state.state == G.STATES.WIN and G.GAME.win then
        return true
    end

    if current_state.state == G.STATES.GAME_OVER then
        return false
    end

    -- -- Auto-skip ROUND_EVAL
    -- if current_state.state == G.STATES.ROUND_EVAL then
    --     local is_busy = false
    --     if G.CONTROLLER and G.CONTROLLER.queue and G.CONTROLLER.locks then
    --         local queue_size = #G.CONTROLLER.queue
    --         local locks_size = 0
    --         for _ in pairs(G.CONTROLLER.locks) do locks_size = locks_size + 1 end

    --         if queue_size > 0 or locks_size > 0 then
    --             is_busy = true
    --         end
    --     end

    --     if is_busy then
    --         return false
    --     end

    --     if #available_actions > 0 and not cash_out_executed then
    --         return true
    --     end
    -- end

    return false
end

--- Execute auto-skip action without AI involvement
--- @param current_state table Current game state
--- @param available_actions table Available actions list  
function AI.execute_auto_skip_action(current_state, available_actions)
    if current_state.state == G.STATES.ROUND_EVAL then
        local current_ante = G.GAME and G.GAME.round_resets and G.GAME.round_resets.ante or 0
        local blind_obj = G.GAME and G.GAME.blind or nil
        local is_boss = blind_obj and blind_obj.boss or false
        local chips_beaten = G.GAME and blind_obj and (G.GAME.chips >= blind_obj.chips) or false

        if current_ante == 9 and is_boss then
            utils.log_ai("Ante 8 boss beaten — restarting run...")
            if G.FUNCS and G.FUNCS.go_to_menu then
                G.FUNCS.go_to_menu()
                G.E_MANAGER:add_event(Event({
                    trigger = 'after', delay = 1.5,
                    func = function()
                        if G.FUNCS.start_run then
                            G.FUNCS.start_run()
                        elseif G.FUNCS.play_new_run then
                            G.FUNCS.play_new_run()
                        end
                        return true
                    end
                }))
            end
            -- state_transition_timer = 30.0
            last_combined_hash = nil
            return
        end
        -- Normal cash out
        local input = require("input")
        local result = input.cash_out()
        if result.success then
            -- utils.log_ai("Blind Beaten!")
            -- utils.log_ai("Cash out successful")
            cash_out_executed = true
            -- state_transition_timer = 0.5
        else
            utils.log_ai("Cash out failed: " .. (result.error or "unknown"))
            last_combined_hash = nil  -- retry next frame
        end
        return
    end
    
    local action_id = available_actions[1]
    utils.log_ai("Auto-executing action: " .. action.get_action_name(action_id))
    
    local exec_params = {}
    --Chnage this to change the seed
    if action_id == 4 then 
        exec_params.seed = "JFKGEEMG"
    end
    local result = action.execute_action(action_id, exec_params)
    if result.success then
        utils.log_ai("Auto-execution successful: " .. action.get_action_name(action_id))
    else
        utils.log_ai("Auto-execution failed: " .. (result.error or "Unknown error"))
    end
    
    -- Force state recheck after auto-execution
    last_combined_hash = nil
end

return AI
