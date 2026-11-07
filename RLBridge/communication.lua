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

--- Writing to a FIFO/socket whose reader died raises SIGPIPE, which kills the process by default.
--- Ignore it explicitly (non-Windows) so the failed write comes back as an error instead.
local sigpipe_ignored = false
local function ignore_sigpipe()
    if sigpipe_ignored or os_name == "Windows_NT" then return end
    sigpipe_ignored = true
    pcall(function()
        local ffi = require("ffi")
        ffi.cdef("typedef void (*rl_sighandler_t)(int); rl_sighandler_t signal(int signum, rl_sighandler_t handler);")
        ffi.C.signal(13, ffi.cast("rl_sighandler_t", 1)) -- SIGPIPE = 13, SIG_IGN = 1
    end)
end

--- Enable communication for a new session
--- Called only when a session is being armed (start key or auto-wake), never while idle, so the
--- game never does a blocking FIFO open without a Python session waiting.
--- @return nil
function COMM.init()
    ignore_sigpipe()
    utils.log_comm("Initializing communication channel. Mode: " .. comm_mode .. " on " .. socket_host .. ":" .. socket_port .. ", OS: " .. os_name)
    comm_enabled = true -- Enable communication, pipes will open on first use
end

--- @return string "socket" or "pipe"
function COMM.get_mode()
    return comm_mode
end

--- Socket mode: try to reach the viewer server (short timeout, keeps the connection on success)
--- @return boolean
function COMM.try_connect()
    return comm_mode == "socket" and COMM.ensure_socket_open()
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
    local conn = sock.tcp()
    conn:settimeout(1) -- a connect must never stall the game for long
    local connected, err = conn:connect(socket_host, socket_port)
    if not connected then
        conn:close()
        conn = nil
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

    -- io.open on a FIFO blocks until the other end opens: never do it unless a live Python is waiting
    -- (covers Python being killed between arming and the first request)
    if not utils.session_ready() then
        utils.log_comm("ERROR: No live Python session (flag missing or stale) — not opening pipes")
        return false
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
--- Any failure (open, write/EPIPE, EOF, bad JSON) returns nil; the caller treats that as a lost
--- connection and stops the session instead of retrying every frame.
--- @param game_state table Current game state data
--- @param available_actions table Available actions list
--- @param stop_requested boolean|nil True on the final request after the start key in toggle mode
--- @return table|nil Action response from AI (action == "SHUTDOWN" when Python is done), nil if error
function COMM.request_action(game_state, available_actions, stop_requested)
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
        stop_requested = stop_requested or nil,
    }

    utils.log_comm(utils.get_timestamp() .. " Sending action request for state: " ..
        tostring(game_state.state) .. " (" .. utils.get_state_name(game_state.state) .. ")" ..
        " | round: " .. tostring(game_state.round_count or 0) ..
        (stop_requested and " | STOP REQUESTED" or ""))

    -- Encode request as JSON
    local json_data = json.encode(request)
    if not json_data then
        utils.log_comm("ERROR: Failed to encode request JSON")
        return nil
    end

    local response_json = nil
    if comm_mode == "socket" then
        local sent, send_err = socket_conn:send(json_data .. "\n")
        if not sent then
            utils.log_comm("ERROR: Socket send failed: " .. tostring(send_err))
            return nil
        end
        -- Python may be busy (e.g. a PPO update) for far longer than the socket timeout: keep waiting
        -- on "timeout"; only a real error or a closed connection counts as a disconnect.
        local buffered = ""
        while true do
            local line, err, partial = socket_conn:receive("*l", buffered)
            if line then
                response_json = line
                break
            end
            if err ~= "timeout" then
                utils.log_comm("ERROR: Socket receive failed: " .. tostring(err))
                return nil
            end
            buffered = partial or buffered
        end
    else
        -- Write request to persistent handle (pcall: a dead reader surfaces as an error, not a crash)
        -- file:write/flush return nil, err on failure (EPIPE) instead of raising; check both
        local call_ok, wrote, write_err = pcall(function()
            local w, werr = request_handle:write(json_data .. "\n")
            if not w then return nil, werr end
            local f, ferr = request_handle:flush() -- Ensure data is sent immediately
            if not f then return nil, ferr end
            return true
        end)
        if not call_ok or not wrote then
            utils.log_comm("ERROR: Failed to write request to pipe: " .. tostring(call_ok and write_err or wrote))
            return nil
        end

        -- Read response from persistent handle (nil = EOF, Python closed or died)
        local read_ok, line = pcall(function() return response_handle:read("*line") end)
        if not read_ok or not line then
            utils.log_comm("ERROR: Failed to read response from pipe (EOF or error)")
            return nil
        end
        response_json = line
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
        pcall(request_handle.close, request_handle)
        request_handle = nil
    end

    if response_handle then
        pcall(response_handle.close, response_handle)
        response_handle = nil
    end

    if socket_conn then
        pcall(socket_conn.close, socket_conn)
        socket_conn = nil
    end

    utils.log_comm("Persistent pipe communication closed")
end

return COMM
