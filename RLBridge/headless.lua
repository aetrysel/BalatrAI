--- RLBridge headless module
--- Skips rendering on headless workers and periodically writes a screenshot + status file for monitoring
---
--- Environment variables:
---   BALATRO_HEADLESS=1          skip love.draw() except on snapshot frames
---   BALATRO_SNAPSHOT_DIR=<dir>  write screen.png and status.json into <dir>
---   BALATRO_SNAPSHOT_SECS=<n>   seconds between snapshots (default 2)

local HEADLESS = {}

HEADLESS.enabled = os.getenv("BALATRO_HEADLESS") == "1"
HEADLESS.snapshot_dir = os.getenv("BALATRO_SNAPSHOT_DIR")
HEADLESS.snapshot_secs = tonumber(os.getenv("BALATRO_SNAPSHOT_SECS")) or 2
HEADLESS.worker_id = os.getenv("BALATRO_WORKER_ID") or "0"

local last_snapshot = -math.huge
local frame_count = 0
local fps_window_start = nil
local fps_window_frames = 0
local fps = 0

--- Write content to path atomically (tmp file + rename) so readers never see a partial file
--- @param path string
--- @param content string
local function write_atomic(path, content)
    local tmp = path .. ".tmp"
    local f = io.open(tmp, "wb")
    if not f then return end
    f:write(content)
    f:close()
    os.rename(tmp, path)
end

--- Snapshot of the run for the monitor page
--- @return table
local function build_status()
    local status = {
        worker_id = HEADLESS.worker_id,
        headless = HEADLESS.enabled,
        time = os.time(),
        frame = frame_count,
        fps = math.floor(fps + 0.5),
        state = G and G.STATE or -1,
    }
    if G and G.GAME then
        status.ante = G.GAME.round_resets and G.GAME.round_resets.ante or 0
        status.round = G.GAME.round or 0
        status.dollars = G.GAME.dollars or 0
        status.chips = G.GAME.chips or 0
        status.blind = G.GAME.blind and G.GAME.blind.name or ""
        status.seed = G.GAME.pseudorandom and G.GAME.pseudorandom.seed or ""
        status.jokers = G.jokers and G.jokers.cards and #G.jokers.cards or 0
    end
    return status
end

local function take_snapshot()
    local dir = HEADLESS.snapshot_dir
    love.graphics.captureScreenshot(function(image_data)
        local ok, file_data = pcall(function() return image_data:encode("png") end)
        if ok and file_data then
            write_atomic(dir .. "/screen.png", file_data:getString())
        end
    end)
    local json = require("dkjson")
    write_atomic(dir .. "/status.json", json.encode(build_status()))
end

--- Called once per frame from love.run in place of the isActive() draw gate
--- @return boolean True if this frame should be drawn and presented
function HEADLESS.should_draw()
    frame_count = frame_count + 1
    local now = love.timer.getTime()
    fps_window_start = fps_window_start or now
    fps_window_frames = fps_window_frames + 1
    if now - fps_window_start >= 1 then
        fps = fps_window_frames / (now - fps_window_start)
        fps_window_start = now
        fps_window_frames = 0
    end

    local snapshot_due = HEADLESS.snapshot_dir and (now - last_snapshot >= HEADLESS.snapshot_secs)
    if snapshot_due then
        last_snapshot = now
        take_snapshot()
        return true
    end
    return not HEADLESS.enabled
end

return HEADLESS
