--- RLBridge communication module
--- Handles dual pipe communication with persistent handles between the game and external AI system

local COMM = {}
local utils = require("utils")
local json = require("dkjson")

-- Dual pipe communication settings with persistent handles
local comm_enabled = false
local request_pipe
local response_pipe

local os_name = os.getenv("OS") or love.system.getOS()

if os_name == "Windows_NT" then
    request_pipe = "\\\\.\\pipe\\balatro_request"
    response_pipe = "\\\\.\\pipe\\balatro_response"
else
    request_pipe = "/tmp/balatro_request"
    response_pipe = "/tmp/balatro_response"
end

local request_handle = nil
local response_handle = nil

-- Communication mode: "pipe" (default) or "socket" (viewer instance)
local comm_mode = os.getenv("BALATRO_COMM_MODE") or "pipe"
local socket_host = os.getenv("BALATRO_SOCKET_HOST") or "127.0.0.1"
local socket_port = tonumber(os.getenv("BALATRO_SOCKET_PORT")) or 9000

local socket_conn = nil  -- LuaSocket connection handle

--- Initialize dual pipe communication with persistent handles
--- Sets up persistent pipe handles with external AI system
--- @return nil
function COMM.init()
    utils.log_comm("Initializing communication channel. Mode: " .. comm_mode .. "on " .. socket_host .. ":" .. socket_port .. ", OS: " .. os_name)
    comm_enabled = true -- Enable communication, pipes will open on first use
end

function COMM.ensure_connection_open()
    if comm_mode == "socket" then
        utils.log_comm("Initializing socket communication on " .. socket_host .. ":" .. socket_port)
        return COMM.ensure_socket_open()
    else
        utils.log_comm("Initializing dual pipe communication...")
        return COMM.ensure_pipes_open()
    end
end

function COMM.ensure_socket_open()
    if socket_conn then return true end

    local sock = require("socket")
    local conn, err = sock.connect(socket_host, socket_port)
    if not conn then
        utils.log_comm("ERROR: Cannot connect to socket: " .. tostring(err))
        return false
    end
    conn:setoption("tcp-nodelay", true)
    conn:settimeout(30) -- Set a timeout for socket operations
    socket_conn = conn
    utils.log_comm("Connected to viewer socket at " .. socket_host .. ":" .. socket_port)
    return true
end

--- Lazy initialization of pipe handles when first needed
--- @return boolean True if pipes are ready, false otherwise
function COMM.ensure_pipes_open()
    if request_handle and response_handle then
        return true -- Already open
    end

    -- Open response pipe for reading (keep open)
    response_handle = io.open(response_pipe, "r")
    if not response_handle then
        utils.log_comm("ERROR: Cannot open response pipe for reading: " .. response_pipe)
        return false
    end

    -- Open request pipe for writing (keep open)
    request_handle = io.open(request_pipe, "w")
    if not request_handle then
        utils.log_comm("ERROR: Cannot open request pipe for writing: " .. request_pipe)
        if response_handle then
            response_handle:close()
            response_handle = nil
        end
        return false
    end

    return true
end

--- Send game turn request to AI and get action via persistent pipe handles
--- @param game_state table Current game state data
--- @param available_actions table Available actions list
--- @return table|nil Action response from AI, nil if error
function COMM.request_action(game_state, available_actions)
    if not comm_enabled then
        utils.log_comm("ERROR: Communication not enabled")
        return nil
    end

    -- Lazy initialization - open pipes when first needed
    if not COMM.ensure_connection_open() then
        utils.log_comm("ERROR: Failed to open communication handles")
        return nil
    end

    local request = {
        game_state = game_state,
        available_actions = available_actions or {},
    }

    utils.log_comm(utils.get_timestamp() .. " Sending action request for state: " ..
        tostring(game_state.state) .. " (" .. utils.get_state_name(game_state.state) .. ")" ..
        " | round: " .. tostring(game_state.round_count or 0))

    -- Encode request as JSON
    local json_data = json.encode(request)
    if not json_data then
        utils.log_comm("ERROR: Failed to encode request JSON")
        return nil
    end
    
    local response_json = nil
    if comm_mode == "socket" then
        socket_conn:send(json_data .. "\n")
        local line, err = socket_conn:receive("*l")
        if not line then
            utils.log_comm("ERROR: Socket receive failed: " .. tostring(err))
            return nil
        end
        response_json = line
    else
        -- Write request to persistent handle
        request_handle:write(json_data .. "\n")
        request_handle:flush() -- Ensure data is sent immediately

        -- Read response from persistent handle
        response_json = response_handle:read("*line")
        if not response_json then
            utils.log_comm("ERROR: Failed to read response from pipe")
            return nil
        end
    end

    if not response_json or response_json == "" then
        utils.log_comm("ERROR: No response received from AI")
        return nil
    end

    local response_data = json.decode(response_json)
    if not response_data then
        utils.log_comm("ERROR: Failed to decode response JSON")
        return nil
    end

    utils.log_comm(utils.get_timestamp() .. " AI action: " .. tostring(response_data.action))
    return response_data
end

--- Check if pipe communication is enabled
--- Returns the current communication status
--- @return boolean True if enabled, false otherwise
function COMM.is_connected()
    return comm_enabled
end

--- Close communication
--- Terminates the persistent pipe handles with the AI system
--- @return nil
function COMM.close()
    comm_enabled = false

    if request_handle then
        request_handle:close()
        request_handle = nil
    end

    if response_handle then
        response_handle:close()
        response_handle = nil
    end

    utils.log_comm("Persistent pipe communication closed")
end

return COMM
