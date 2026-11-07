--- RLBridge utility module
--- Common helper functions for debugging, table operations, and state management
--- Provides tools for development and troubleshooting

local UTIL = {}
UTIL.completed_success_msg = "completed successfully"

-- Logging Functions
--- Log with AI prefix
--- @param msg string Message to log
UTIL.log_ai = function(msg)
    print("[AI] " .. msg)
end

--- Log with AI Dummy prefix (for auto-play testing)
--- @param msg string Message to log
UTIL.log_ai_dummy = function(msg)
    print("[AI DUMMY] " .. msg)
end

--- Log with communication prefix
--- @param msg string Message to log
UTIL.log_comm = function(msg)
    print("[COMM] " .. msg)
end

--- Log with input prefix
--- @param msg string Message to log
UTIL.log_input = function(msg)
    print("[INPUT] " .. msg)
end

--- Log with error prefix
--- @param msg string Message to log
UTIL.log_error = function(msg)
    print("[ERROR] " .. msg)
end

--- Log with testing prefix
--- @param msg string Message to log
UTIL.log_testing = function(msg)
    print("[TESTING] " .. msg)
end

--- Convert state ID to readable name
--- Maps numeric state IDs to their string representations for debugging
--- @param state_id number Numeric state identifier
--- @return string Human-readable state name or "UNKNOWN_STATE"
function UTIL.get_state_name(state_id)
    for key, value in pairs(G.STATES) do
        if value == state_id then
            return key
        end
    end
    return "UNKNOWN_STATE"
end

--- Simple table dump for debugging
--- Prints all key-value pairs in a table
--- @param table table Table to dump
--- @return nil
function UTIL.dump_table(table)
    for k, v in pairs(table) do
        print(k, v)
    end
end

--- Deep table dump with recursion protection
--- Recursively prints table contents with proper indentation
--- @param tbl table Table to dump
--- @param indent number|nil Current indentation level
--- @param visited table|nil Recursion tracking table
--- @return string Formatted table contents
function UTIL.dump_everything(tbl, indent, visited)
    indent = indent or 0
    visited = visited or {}

    local prefix = string.rep("  ", indent)
    local output = {}

    if visited[tbl] then
        table.insert(output, prefix .. "*RECURSION*")
        return table.concat(output, "\n")
    end

    visited[tbl] = true

    if type(tbl) ~= "table" then
        return tostring(tbl)
    end

    local keys = {}
    for k in pairs(tbl) do table.insert(keys, k) end
    table.sort(keys, function(a, b) return tostring(a) < tostring(b) end)

    for _, k in ipairs(keys) do
        local v = tbl[k]
        local key = tostring(k)
        local value_type = type(v)

        if value_type == "table" then
            table.insert(output, prefix .. key .. " = {")
            table.insert(output, UTIL.dump_everything(v, indent + 1, visited))
            table.insert(output, prefix .. "}")
        elseif value_type == "function" then
            table.insert(output, prefix .. key .. " = <function>")
        elseif value_type == "userdata" then
            table.insert(output, prefix .. key .. " = <userdata>")
        elseif value_type == "thread" then
            table.insert(output, prefix .. key .. " = <thread>")
        else
            local val_str = tostring(v)
            if value_type == "string" then
                val_str = '"' .. val_str .. '"'
            end
            table.insert(output, prefix .. key .. " = " .. val_str)
        end
    end

    return table.concat(output, "\n")
end

--- Helper function to get action names for logging
--- Converts action IDs to readable names for debug output
--- @param available_actions table Available actions table with IDs
--- @return table Array of action names
function UTIL.get_action_names(available_actions)
    local action = require("actions")
    local names = {}
    for _, action_id in ipairs(available_actions) do
        table.insert(names, action.get_action_name(action_id))
    end
    return names
end

function UTIL.contains(tbl, val)
    for x, _ in ipairs(tbl) do
        if tbl[x] == val then
            return true
        end
    end
    return false
end

--- Get current timestamp as formatted string
--- Returns current time in HH:MM:SS.mmm format for debugging
--- @return string Formatted timestamp
function UTIL.get_timestamp()
    local time = os.time()
    local ms = (os.clock() % 1) * 1000
    return os.date("%H:%M:%S", time) .. string.format(".%03d", ms)
end

--- Check if the game is still settling a previous action and would ignore button presses
--- Same guard the game puts on its Use/Sell/Skip buttons (controller locks or a pending stop_use)
--- @return boolean True if inputs are currently blocked
function UTIL.is_input_blocked()
    return (G.CONTROLLER and (G.CONTROLLER.locked or G.CONTROLLER.locks.frame))
        or (G.GAME and (G.GAME.STOP_USE or 0) > 0)
        or false
end

--- Check if the game's own UI button for a card would be clickable right now
--- Runs the button's G.FUNCS state callback (e.g. 'can_buy_and_use') against a stub UI element,
--- so the agent is held to exactly the rules a human player sees, including SMODS/mod overrides
--- @param func_name string Name of the G.FUNCS button-state callback
--- @param card table Card the button belongs to
--- @return boolean True if the button would be enabled
function UTIL.is_button_enabled(func_name, card)
    local func = G.FUNCS and G.FUNCS[func_name]
    if not func or not card then return false end
    local e = { config = { ref_table = card }, UIBox = { states = {} } }
    local ok = pcall(func, e)
    return ok and e.config.button ~= nil
end

--- Check if a booster pack card can be taken right now (the pack's USE / SELECT button is enabled)
--- Consumables get USE (can_use_consumeable), or SELECT when a modded pack saves them instead;
--- jokers and playing cards get SELECT (can_select_card)
--- @param card table Card in G.pack_cards
--- @return boolean True if the card's button would be enabled
function UTIL.can_take_pack_card(card)
    if not card or not card.ability then return false end
    local func_name = 'can_select_card'
    if card.ability.consumeable then
        func_name = 'can_use_consumeable'
        local ok, selectable = pcall(function()
            return booster_obj and SMODS and SMODS.card_select_area
                and SMODS.card_select_area(card, booster_obj) and card:selectable_from_pack(booster_obj)
        end)
        if ok and selectable then
            func_name = 'can_select_from_booster'
        end
    end
    return UTIL.is_button_enabled(func_name, card)
end

--- Check if there is room for a card bought from the shop
--- Runs the game's check_for_buy_space with its "no space" popup silenced for the call
--- @param card table Card in a shop area
--- @return boolean True if the bought card would have somewhere to go
function UTIL.has_buy_space(card)
    if not card or not G.FUNCS or not G.FUNCS.check_for_buy_space then return false end
    local alert = alert_no_space
    alert_no_space = function() end
    local ok, has_space = pcall(G.FUNCS.check_for_buy_space, card)
    alert_no_space = alert
    return ok and has_space == true
end

--- Check if the game is ready to accept actions (no animations/blocking events)
--- Uses Balatro's event manager to determine if animations are playing
--- @return boolean True if ready for actions, false if animations are playing
function UTIL.is_game_ready_for_action()
    -- Check if game is paused
    if G.SETTINGS and G.SETTINGS.paused then
        return false
    end

    -- Check if the game is still settling a previous action (its buttons are disabled)
    if UTIL.is_input_blocked() then
        return false
    end

    -- Check if there are blocking events in the queue
    if G.E_MANAGER and G.E_MANAGER.queues then
        for queue_name, queue in pairs(G.E_MANAGER.queues) do
            for _, event in ipairs(queue) do
                if event.blocking and not event.complete then
                    return false
                end
            end
        end
    end

    -- Check if state transition is complete
    if G.STATE_COMPLETE == false then
        return false
    end

    return true
end

--- Check if ALL events in the queue are complete (including non-blocking)
--- Used in pack states where non-blocking animation events hold card object references
--- that become nil after Handy removes animations, causing crashes if actions fire early
--- @return boolean True if all events are complete or queue is empty
function UTIL.is_event_queue_empty()
    if not G.E_MANAGER or not G.E_MANAGER.queues then return true end
    for _, queue in pairs(G.E_MANAGER.queues) do
        for _, event in ipairs(queue) do
            if not event.complete then
                return false
            end
        end
    end
    return true
end

local run_config_cache = nil

--- Read the run config written by train_balatro.py ({seed, deck, endless})
--- Cached after the first read; pass reload=true to re-read the file (done on each START_RUN)
--- @param reload boolean|nil Force a re-read from disk
--- @return table Config table (empty if the file is missing or invalid)
function UTIL.get_run_config(reload)
    if run_config_cache and not reload then
        return run_config_cache
    end
    local paths = { "/tmp/balatro_run_config.json" }
    local appdata = os.getenv("APPDATA")
    if appdata then
        table.insert(paths, 1, appdata .. "/Balatro/Mods/RLBridge/run_config.json")
    end
    run_config_cache = {}
    for _, path in ipairs(paths) do
        local f = io.open(path, "r")
        if f then
            local dkjson = require("dkjson")
            local config = dkjson.decode(f:read("*all"))
            f:close()
            if type(config) == "table" then
                run_config_cache = config
                break
            end
        end
    end
    return run_config_cache
end

--- True when the ante 8 boss has just been defeated (same condition output.lua reports as game_win)
--- @return boolean
function UTIL.is_game_won_round_eval()
    if G.STATE ~= G.STATES.ROUND_EVAL then
        return false
    end
    local ante = G.GAME and G.GAME.round_resets and G.GAME.round_resets.ante or 0
    local blind_states = G.GAME and G.GAME.round_resets and G.GAME.round_resets.blind_states
    return ante == 9 and blind_states and blind_states.Boss == "Defeated" or false
end

-- ===== Persistent bridge settings + session-ready flag (shared with ai/utils/lifecycle.py) =====

local DEFAULT_SETTINGS = {
    start_key = "ctrl+r",
    start_key_mode = "arm",
    auto_wake = true,
    win_hold_secs = 60,
    win_hold_secs_endless = 15,
}

--- Path of rlbridge_settings.json: Balatro's LOVE save directory (Linux ~/.local/share/love/Balatro,
--- Windows %APPDATA%/Balatro; same folder as lovely.log). Persistent and outside the repo, so git never
--- tracks it. BALATRO_SETTINGS_PATH overrides it (Python honours the same variable).
--- @return string
function UTIL.get_settings_path()
    local override = os.getenv("BALATRO_SETTINGS_PATH")
    if override and override ~= "" then
        return override
    end
    local dir
    if love and love.filesystem and love.filesystem.getSaveDirectory then
        local ok, result = pcall(love.filesystem.getSaveDirectory)
        if ok and type(result) == "string" and result ~= "" then
            dir = result
        end
    end
    if not dir then
        local appdata = os.getenv("APPDATA")
        if appdata then
            dir = appdata .. "/Balatro"
        else
            dir = (os.getenv("XDG_DATA_HOME") or ((os.getenv("HOME") or "") .. "/.local/share")) .. "/love/Balatro"
        end
    end
    return dir .. "/rlbridge_settings.json"
end

--- Read rlbridge_settings.json and merge it over the defaults (invalid or missing values fall back)
--- @return table settings
function UTIL.load_settings()
    local settings = {}
    for k, v in pairs(DEFAULT_SETTINGS) do settings[k] = v end
    local f = io.open(UTIL.get_settings_path(), "r")
    if not f then
        return settings
    end
    local ok, data = pcall(function() return require("dkjson").decode(f:read("*all")) end)
    f:close()
    if not ok or type(data) ~= "table" then
        return settings
    end
    if type(data.start_key) == "string" and UTIL.parse_keybind(data.start_key) then
        settings.start_key = data.start_key
    end
    if data.start_key_mode == "arm" or data.start_key_mode == "toggle" then
        settings.start_key_mode = data.start_key_mode
    end
    if type(data.auto_wake) == "boolean" then
        settings.auto_wake = data.auto_wake
    end
    for _, name in ipairs({ "win_hold_secs", "win_hold_secs_endless" }) do
        if type(data[name]) == "number" and data[name] >= 0 then
            settings[name] = data[name]
        end
    end
    return settings
end

--- Parse "ctrl+shift+t" / "f9" into { key = "t", ctrl = true, shift = true, alt = false }
--- Modifiers: ctrl, shift, alt (left or right physical key). Exactly one non-modifier key.
--- @param text string
--- @return table|nil Parsed keybind, nil if the format is invalid
function UTIL.parse_keybind(text)
    if type(text) ~= "string" then return nil end
    local tokens = {}
    for token in (text:lower() .. "+"):gmatch("([^+]*)%+") do
        token = token:gsub("^%s+", ""):gsub("%s+$", "")
        if token == "" then return nil end
        tokens[#tokens + 1] = token
    end
    local parsed = { ctrl = false, shift = false, alt = false }
    local key = table.remove(tokens)
    for _, mod in ipairs(tokens) do
        if mod == "control" then mod = "ctrl" end
        if mod ~= "ctrl" and mod ~= "shift" and mod ~= "alt" then return nil end
        if parsed[mod] then return nil end
        parsed[mod] = true
    end
    if key == "enter" then key = "return" end -- LOVE calls it "return"
    if key == "ctrl" or key == "shift" or key == "alt" or key == "control" then return nil end
    parsed.key = key
    return parsed
end

--- True if the pressed love key plus the currently held modifiers exactly match the parsed keybind
--- @param parsed table Result of parse_keybind
--- @param key string love.keypressed key name
--- @return boolean
function UTIL.keybind_matches(parsed, key)
    if not parsed or key ~= parsed.key then return false end
    local kb = love.keyboard
    local ctrl = kb.isDown("lctrl", "rctrl")
    local shift = kb.isDown("lshift", "rshift")
    local alt = kb.isDown("lalt", "ralt")
    return ctrl == parsed.ctrl and shift == parsed.shift and alt == parsed.alt
end

--- Path of the session-ready flag Python writes after creating the FIFOs (a plain file, so checking
--- it can never block). BALATRO_SESSION_FLAG overrides it (parallel runs use one per run dir).
--- @return string
function UTIL.get_session_flag_path()
    local override = os.getenv("BALATRO_SESSION_FLAG")
    if override and override ~= "" then
        return override
    end
    local appdata = os.getenv("APPDATA")
    if appdata then
        return appdata .. "/Balatro/Mods/RLBridge/session_ready"
    end
    return "/tmp/balatro_session_ready"
end

--- Is a process alive? kill(pid, 0) via FFI (Linux and macOS); EPERM still means it exists.
--- @return boolean|nil nil when it cannot be determined (Windows or no FFI), so the caller trusts the flag
local kill_declared = false
local function os_pid_alive(pid)
    if (os.getenv("OS") or "") == "Windows_NT" then return nil end
    local ok, result = pcall(function()
        local ffi = require("ffi")
        if not kill_declared then
            ffi.cdef("int kill(int pid, int sig);")
            kill_declared = true
        end
        if ffi.C.kill(pid, 0) == 0 then return true end
        return ffi.errno() == 1 -- EPERM
    end)
    if ok then return result end
    return nil
end

--- True when a Python session is waiting: flag file present and its PID alive (when that can be checked).
--- The PID check ignores a stale flag left by a killed Python, which would otherwise freeze the
--- game in a blocking FIFO open.
--- @return boolean
function UTIL.session_ready()
    local f = io.open(UTIL.get_session_flag_path(), "r")
    if not f then
        return false
    end
    local pid = tonumber((f:read("*line") or ""):match("%d+"))
    f:close()
    if pid and os_pid_alive(pid) == false then
        return false
    end
    return true
end

return UTIL
