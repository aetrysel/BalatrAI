--- RLBridge initialization module
--- Handles initial game setup and configuration for the RL system
--- Sets up animation skipping and starts a new Balatro run with AI integration

local INIT = {}

-- BALATRO_GAMESPEED / BALATRO_ANIMATIONS / BALATRO_FPS_CAP let a launcher run a normal-speed viewer next to fast workers
DISABLE_ANIMATIONS = os.getenv("BALATRO_ANIMATIONS") ~= "1"
G.SETTINGS.skip_splash = 'Yes'
G.SETTINGS.GAMESPEED = tonumber(os.getenv("BALATRO_GAMESPEED")) or 100
G.SETTINGS.reduced_motion = DISABLE_ANIMATIONS
G.FPS_CAP = tonumber(os.getenv("BALATRO_FPS_CAP")) or G.FPS_CAP

-- Headless workers have no one listening
if os.getenv("BALATRO_HEADLESS") == "1" then
    G.SETTINGS.SOUND.volume = 0
end

--- Should be called after G:start_up() to safely initialize and start a run
--- Initializes the AI system and starts a new Balatro run with default settings
--- @return nil
function INIT.start_run()
    print("Starting balatro run")

    
    -- Initialize AI system
    local ai = require("ai")
    ai.init()
end

return INIT
