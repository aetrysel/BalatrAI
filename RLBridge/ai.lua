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
local start_key_pressed = false
local cash_out_executed = false
local retry_count = 0

local blind_forced = false
local blind_force_timer = 0
local win_overlay_handled = false

-- Lifecycle: settings come from rlbridge_settings.json (re-read on each wake/arm and idle tick)
local settings = utils.load_settings()
local stop_requested = false        -- toggle mode: next request carries stop_requested = true
local last_wake_check = 0           -- love.timer.getTime() of the last idle auto-wake poll
local wake_blocked_until = 0        -- no auto-wake right after a stop, so a dying session's flag can't re-arm us
local WAKE_POLL_SECS = 1.0
local SOCKET_WAKE_POLL_SECS = 3.0   -- socket mode has no flag file: probe the viewer server instead
local WAKE_COOLDOWN_SECS = 3.0

-- Win hold: freezes the AI (no requests) for a fixed real-time duration, then runs the end action.
-- nil when not holding; otherwise { start, duration, endless, source }
local win_hold = nil
local test_win_hold_done = false

local SHOP_DEBUG = false -- Set to true to enable shop state debug logging

local function is_pack_state(state)
    return state == G.STATES.TAROT_PACK
        or state == G.STATES.PLANET_PACK
        or state == G.STATES.SPECTRAL_PACK
        or state == G.STATES.STANDARD_PACK
        or state == G.STATES.BUFFOON_PACK
        or (G.STATES.SMODS_BOOSTER_OPENED ~= nil and state == G.STATES.SMODS_BOOSTER_OPENED)
end

--- Arm a session: enable communication and start answering requests
--- @param reason string Logged
local function arm_session(reason)
    settings = utils.load_settings()
    communication.init()
    rl_training_active = true
    stop_requested = false
    pending_action = nil
    last_combined_hash = nil
    utils.log_ai("\n\nRL Training STARTED (" .. reason .. ")")
end

--- Stop the session and return the game to the main menu, where it idles until the next wake
--- Triggers: SHUTDOWN response, lost connection (nil/EOF/write error), start key in toggle mode.
--- @param reason string Logged
function AI.stop_session(reason)
    utils.log_ai("RL Training STOPPED (" .. tostring(reason) .. ") — returning to main menu")
    if G.OVERLAY_MENU and G.FUNCS.exit_overlay_menu then
        pcall(G.FUNCS.exit_overlay_menu)
    end
    if G.STAGE == G.STAGES.RUN and G.FUNCS.go_to_menu then
        local ok, err = pcall(G.FUNCS.go_to_menu)
        if not ok then
            utils.log_ai("go_to_menu failed: " .. tostring(err))
        end
    end
    rl_training_active = false
    communication.close()
    pending_action = nil
    last_combined_hash = nil
    win_hold = nil
    stop_requested = false
    retry_count = 0
    wake_blocked_until = love.timer.getTime() + WAKE_COOLDOWN_SECS
end

--- Freeze the AI for the win screen, then return to normal play (real time, game speed independent)
--- Every win path calls this; a second call while holding is ignored.
--- Duration comes from win_hold_secs / win_hold_secs_endless in rlbridge_settings.json.
--- When it ends: endless dismisses the overlay and keeps playing; otherwise go to the menu so the
--- START_RUN auto-skip begins the next run. The "test" source (BALATRO_TEST_WIN_HOLD) just resumes.
--- @param source string Which win path triggered it (logged): overlay, state_win, cash_out, boss, test
function AI.begin_win_hold(source)
    if win_hold then
        return
    end
    local endless = utils.get_run_config().endless and true or false
    settings = utils.load_settings()
    local duration = endless and settings.win_hold_secs_endless or settings.win_hold_secs
    win_hold = { start = love.timer.getTime(), duration = duration, endless = endless, source = source }
    pending_action = nil
    last_combined_hash = nil
    utils.log_ai(string.format("Win hold started (%s): %ss%s", source, tostring(duration), endless and " (endless)" or ""))
end

local function end_win_hold()
    local hold = win_hold
    win_hold = nil
    last_combined_hash = nil
    utils.log_ai(string.format("Win hold finished (%s) after %.1fs", hold.source, love.timer.getTime() - hold.start))
    if hold.source == "test" then
        return
    end
    if G.OVERLAY_MENU then
        G.FUNCS.exit_overlay_menu()
    end
    if hold.endless then
        utils.log_ai("Continuing in endless mode")
    else
        G.FUNCS.go_to_menu()
    end
end

--- Initialize AI system
--- Installs hooks only; communication stays off until a session is armed
--- @return nil
function AI.init()
    require("stats").install()
    require("joker_tracker").install()

    -- Hook into Love2D keyboard events
    if love and love.keypressed then
        local original_keypressed = love.keypressed
        love.keypressed = function(key, scancode, isrepeat)
            if not isrepeat and utils.keybind_matches(utils.parse_keybind(settings.start_key), key) then
                start_key_pressed = true
            end

            -- Call original function
            if original_keypressed then
                original_keypressed(key, scancode, isrepeat)
            end
        end
    end
end

--- Handle a start key press: arm/wake when idle; in toggle mode request a clean stop when active
local function handle_start_key()
    if rl_training_active then
        if settings.start_key_mode == "toggle" then
            stop_requested = true
            last_combined_hash = nil -- force a request as soon as actions are available
            utils.log_ai("Stop requested via start key — stopping at the next request")
        end
        return
    end
    settings = utils.load_settings()
    if communication.get_mode() ~= "socket" and not utils.session_ready() then
        -- A blocking FIFO open with no Python would freeze the game (a socket connect cannot block)
        utils.log_ai("Start key pressed but no Python session is waiting (start train_balatro first)")
        return
    end
    arm_session("start key")
end

--- Main AI update loop (called every frame)
--- Monitors game state changes, handles communication, and executes AI actions
--- @return nil
function AI.update()
    if start_key_pressed then
        start_key_pressed = false
        handle_start_key()
    end

    -- Inactive: stay completely quiet (no auto-skip, no FIFO opens) apart from the cheap auto-wake poll
    if not rl_training_active then
        local now = love.timer.getTime()
        local socket_mode = communication.get_mode() == "socket"
        if now - last_wake_check >= (socket_mode and SOCKET_WAKE_POLL_SECS or WAKE_POLL_SECS) then
            last_wake_check = now
            settings = utils.load_settings()
            if settings.auto_wake and now >= wake_blocked_until then
                local ready = socket_mode and communication.try_connect() or (not socket_mode and utils.session_ready())
                if ready then
                    arm_session("auto-wake: session ready")
                end
            end
        end
        return
    end

    if win_hold then
        if love.timer.getTime() - win_hold.start >= win_hold.duration then
            end_win_hold()
        end
        return
    end

    -- Debug: exercise the win hold without a real win (first ROUND_EVAL of the session)
    if os.getenv("BALATRO_TEST_WIN_HOLD") == "1" and not test_win_hold_done and G.STATE == G.STATES.ROUND_EVAL then
        test_win_hold_done = true
        utils.log_ai("TEST WIN HOLD")
        AI.begin_win_hold("test")
        return
    end

    -- Win overlay ("You Win!"): hold the win screen, then endless dismisses it and keeps playing, otherwise new run
    if G.STATE == G.STATES.MENU then
        win_overlay_handled = false
    elseif G.GAME and G.GAME.won and G.OVERLAY_MENU and not win_overlay_handled then
        win_overlay_handled = true
        AI.begin_win_hold("overlay")
        return
    end
    -- Allow pack states through even if controller is locked (card deal animations set locking_buttons)
    if not is_pack_state(G.STATE) and (G.CONTROLLER.locking_buttons or G.CONTROLLER.interrupt_inputs or G.CONTROLLER.locks[1]) then
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
        local stopping = stop_requested
        stop_requested = false
        local ai_response = communication.request_action(current_state, available_actions, stopping)

        if not ai_response then
            -- nil = could not open, write failed or EOF: Python is gone. Stop instead of retrying every frame.
            AI.stop_session("disconnected")
            return
        end
        if ai_response.action == "SHUTDOWN" then
            AI.stop_session(stopping and "stop key" or "shutdown")
            return
        end
        pending_action = ai_response
        last_combined_hash = combined_hash
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
            -- Local copy: executing may call begin_win_hold, which clears pending_action
            local executing = pending_action
            local exec_params = executing.params or {}
            if executing.seed then
                exec_params.seed = executing.seed
            end
            if executing.deck then
                exec_params.deck = executing.deck
            end
            local result = action.execute_action(executing.action, exec_params)
            if result.success then
                utils.log_ai("Action executed successfully: " .. executing.action)
                retry_count = 0
                pending_action = nil
                -- Re-request even if the action left the observed state unchanged
                -- (e.g. re-selecting the same hand cards inside a pack)
                last_combined_hash = nil
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

    --- ===== OLD ROUND EVAL CODE ====
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

    -- Python handles ALL ROUND_EVAL states including game win detection
    if current_state.state == G.STATES.ROUND_EVAL then
        return false
    end

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
            utils.log_ai("Ante 8 boss beaten")
            AI.begin_win_hold("boss")
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
    
    -- WIN state: hold the win screen, then the next run starts
    if current_state.state == G.STATES.WIN then
        AI.begin_win_hold("state_win")
        return
    end

    local action_id = available_actions[1]
    utils.log_ai("Auto-executing action: " .. action.get_action_name(action_id))
    
    local exec_params = {}
    if action_id == 4 then
        local config = utils.get_run_config(true)
        if config then
            exec_params.seed = config.seed
            exec_params.deck = config.deck
        end
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
