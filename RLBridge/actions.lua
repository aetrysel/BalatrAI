--- RLBridge action registry module
--- Defines available AI actions, their conditions, and execution logic

local input = require("input")
local utils = require("utils")
local ACTIONS = {}

-- Core gameplay actions
ACTIONS.SELECT_HAND      = 1
ACTIONS.PLAY_HAND        = 2
ACTIONS.DISCARD_HAND     = 3

ACTIONS.START_RUN        = 4
ACTIONS.SELECT_BLIND     = 5
ACTIONS.RESTART_RUN      = 6

ACTIONS.BUY_JOKER        = 7
ACTIONS.SELL_JOKER       = 8
ACTIONS.REROLL_SHOP      = 9
ACTIONS.SKIP_SHOP        = 10
ACTIONS.CASH_OUT         = 11

-- Extended actions
ACTIONS.BUY_VOUCHER      = 12
ACTIONS.BUY_BOOSTER      = 13
ACTIONS.USE_CONSUMABLE   = 14
ACTIONS.SELECT_PACK_CARD = 15
ACTIONS.SKIP_BOOSTER     = 16
ACTIONS.SKIP_BLIND       = 17

local ACTION_IDS = {
    select_hand      = ACTIONS.SELECT_HAND,
    play_hand        = ACTIONS.PLAY_HAND,
    discard_hand     = ACTIONS.DISCARD_HAND,
    start_run        = ACTIONS.START_RUN,
    select_blind     = ACTIONS.SELECT_BLIND,
    restart_run      = ACTIONS.RESTART_RUN,
    buy_joker        = ACTIONS.BUY_JOKER,
    sell_joker       = ACTIONS.SELL_JOKER,
    reroll_shop      = ACTIONS.REROLL_SHOP,
    skip_shop        = ACTIONS.SKIP_SHOP,
    cash_out         = ACTIONS.CASH_OUT,
    buy_voucher      = ACTIONS.BUY_VOUCHER,
    buy_booster      = ACTIONS.BUY_BOOSTER,
    use_consumable   = ACTIONS.USE_CONSUMABLE,
    select_pack_card = ACTIONS.SELECT_PACK_CARD,
    skip_booster     = ACTIONS.SKIP_BOOSTER,
    skip_blind       = ACTIONS.SKIP_BLIND,
}

local ID_TO_ACTION = {
    [ACTIONS.SELECT_HAND]      = "select_hand",
    [ACTIONS.PLAY_HAND]        = "play_hand",
    [ACTIONS.DISCARD_HAND]     = "discard_hand",
    [ACTIONS.START_RUN]        = "start_run",
    [ACTIONS.SELECT_BLIND]     = "select_blind",
    [ACTIONS.RESTART_RUN]      = "restart_run",
    [ACTIONS.BUY_JOKER]        = "buy_joker",
    [ACTIONS.SELL_JOKER]       = "sell_joker",
    [ACTIONS.REROLL_SHOP]      = "reroll_shop",
    [ACTIONS.SKIP_SHOP]        = "skip_shop",
    [ACTIONS.CASH_OUT]         = "cash_out",
    [ACTIONS.BUY_VOUCHER]      = "buy_voucher",
    [ACTIONS.BUY_BOOSTER]      = "buy_booster",
    [ACTIONS.USE_CONSUMABLE]   = "use_consumable",
    [ACTIONS.SELECT_PACK_CARD] = "select_pack_card",
    [ACTIONS.SKIP_BOOSTER]     = "skip_booster",
    [ACTIONS.SKIP_BLIND]       = "skip_blind",
}

local function is_pack_state()
    return G.STATE == G.STATES.TAROT_PACK
        or G.STATE == G.STATES.PLANET_PACK
        or G.STATE == G.STATES.SPECTRAL_PACK
        or G.STATE == G.STATES.STANDARD_PACK
        or G.STATE == G.STATES.BUFFOON_PACK
end

local action_state = {}

function ACTIONS.reset_state()
    action_state = {}
end

function ACTIONS.mark_executed(action_name)
    action_state[action_name] = true
end

local action_registry = {
    start_run = {
        execute = function(params)
            return input.start_run(params)
        end,
        available_when = function()
            return G.STATE == G.STATES.MENU and not action_state.start_run
        end,
    },
    select_blind = {
        execute = function(params)
            return input.select_blind()
        end,
        available_when = function()
            return G.STATE == G.STATES.BLIND_SELECT and G.GAME.blind_on_deck ~= nil
                and not action_state.select_blind and not action_state.skip_blind
                and utils.is_game_ready_for_action()
        end,
    },
    skip_blind = {
        execute = function(params)
            return input.skip_blind()
        end,
        available_when = function()
            return G.STATE == G.STATES.BLIND_SELECT
                and G.GAME.blind_on_deck ~= nil
                and G.GAME.blind_on_deck ~= 'Boss'
                and not action_state.skip_blind
                and utils.is_game_ready_for_action()
        end,
    },
    select_hand = {
        execute = function(params)
            local boss_name = G.GAME and G.GAME.blind and G.GAME.blind.name or nil
            print(boss_name)
            return input.select_hand(params, boss_name)
        end,
        available_when = function()
            return G.STATE == G.STATES.SELECTING_HAND and G.hand and G.hand.cards and #G.hand.highlighted <= 0 and
                utils.is_game_ready_for_action() and
                not action_state.select_hand
        end,
    },
    play_hand = {
        execute = function(params)
            ACTIONS.reset_state()
            return input.play_hand()
        end,
        available_when = function()
            return G.STATE == G.STATES.SELECTING_HAND and G.hand and G.hand.cards and
                #G.hand.highlighted > 0 and utils.is_game_ready_for_action() and
                G.GAME.current_round.hands_left > 0 and
                not action_state.play_hand
        end,
    },
    discard_hand = {
        execute = function(params)
            ACTIONS.reset_state()
            return input.discard_hand()
        end,
        available_when = function()
            return G.STATE == G.STATES.SELECTING_HAND and G.hand and G.hand.cards and
                #G.hand.highlighted > 0 and utils.is_game_ready_for_action() and
                G.GAME.current_round.discards_left > 0 and
                not action_state.discard_hand
        end,
    },
    restart_run = {
        execute = function(params)
            utils.log_input("restart_run execute called, params: " .. tostring(params))
            if params then
                utils.log_input("params.seed = " .. tostring(params.seed))
            end
            return input.start_run(params)
        end,
        available_when = function()
            return (G.STATE == G.STATES.GAME_OVER) and not action_state.restart_run
        end,
    },
    -- Buys any item from G.shop_jokers by slot (jokers, tarots, planets that appear there)
    buy_joker = {
        execute = function(params)
            local slot = params and params[1] or 1
            return input.buy_shop_joker_slot(slot)
        end,
        available_when = function()
            return G.STATE == G.STATES.SHOP
                and G.shop_jokers
                and G.shop_jokers.cards
                and #G.shop_jokers.cards > 0
        end,
    },
    sell_joker = {
        execute = function(params)
            local slot = params and params[1] or 1
            return input.sell_joker(slot)
        end,
        available_when = function()
            return G.STATE == G.STATES.SHOP
                and G.jokers and #G.jokers.cards > 0
        end,
    },
    reroll_shop = {
        execute = function(params)
            return input.reroll_shop()
        end,
        available_when = function()
            return G.STATE == G.STATES.SHOP
                and G.GAME.dollars >= G.GAME.current_round.reroll_cost
        end,
    },
    skip_shop = {
        execute = function(params)
            return input.skip_shop()
        end,
        available_when = function()
            return G.STATE == G.STATES.SHOP
        end,
    },
    cash_out = {
        execute = function(params)
            utils.log_ai("ROUND_EVAL value = " .. tostring(G.STATES.ROUND_EVAL))
            utils.log_ai("Current G.STATE = " .. tostring(G.STATE))
            return input.cash_out()
        end,
        available_when = function()
            return G.STATE == G.STATES.ROUND_EVAL and not action_state.cash_out and utils.is_game_ready_for_action()
        end,
    },
    buy_voucher = {
        execute = function(params)
            return input.buy_voucher()
        end,
        available_when = function()
            if G.STATE ~= G.STATES.SHOP then return false end
            if not G.shop_vouchers or not G.shop_vouchers.cards or #G.shop_vouchers.cards == 0 then return false end
            local card = G.shop_vouchers.cards[1]
            return card and G.GAME.dollars >= (card.cost or 10)
        end,
    },
    buy_booster = {
        execute = function(params)
            local slot = params and params[1] or 1
            return input.buy_booster(slot)
        end,
        available_when = function()
            if G.STATE ~= G.STATES.SHOP then return false end
            return G.shop_booster and G.shop_booster.cards and #G.shop_booster.cards > 0
        end,
    },
    use_consumable = {
        execute = function(params)
            local slot = params and params[1] or 1
            return input.use_consumable(slot)
        end,
        available_when = function()
            return G.STATE == G.STATES.SELECTING_HAND
                and G.consumeables and G.consumeables.cards and #G.consumeables.cards > 0
                and utils.is_game_ready_for_action()
                and not action_state.use_consumable
        end,
    },
    select_pack_card = {
        execute = function(params)
            local slot = params and params[1] or 1
            return input.select_pack_card(slot)
        end,
        available_when = function()
            return is_pack_state()
                and G.pack_cards and G.pack_cards.cards and #G.pack_cards.cards > 0
                and (G.GAME.pack_choices or 0) > 0
                and not action_state.select_pack_card
        end,
    },
    skip_booster = {
        execute = function(params)
            ACTIONS.reset_state()
            return input.skip_booster()
        end,
        available_when = function()
            return is_pack_state()
        end,
    },
}

function ACTIONS.get_available_actions()
    local available = {}
    for action_name, action_def in pairs(action_registry) do
        if action_def.available_when() then
            table.insert(available, ACTIONS.get_action_id(action_name))
        end
    end
    return available
end

function ACTIONS.is_action_available(action_name)
    local action_def = action_registry[action_name]
    return action_def and action_def.available_when()
end

function ACTIONS.execute_action(action_id, params)
    local action_name = ID_TO_ACTION[action_id]
    if not action_name then
        utils.log_input("ERROR: Invalid action ID: " .. tostring(action_id))
        return { success = false, error = "Invalid action ID" }
    end
    if not params then
        utils.log_input("ERROR: Invalid Params: " .. tostring(params))
        return { success = false, error = "Invalid params" }
    end

    utils.log_input("Executing action: " .. action_name)

    if not ACTIONS.is_action_available(action_name) then
        utils.log_input("ERROR: Action not available: " .. action_name)
        return { success = false, error = "Action not available" }
    end

    local action_def = action_registry[action_name]
    if action_def and action_def.execute then
        local result = action_def.execute(params)
        if result.success then
            if action_name == "play_hand" or action_name == "discard_hand" or action_name == "skip_booster" then
                ACTIONS.reset_state()
            else
                ACTIONS.mark_executed(action_name)
            end
        end
        return result
    end
    return { success = false, error = "Unknown action" }
end

function ACTIONS.get_action_name(action_id)
    return ID_TO_ACTION[action_id]
end

function ACTIONS.get_action_id(action_name)
    return ACTION_IDS[action_name]
end

return ACTIONS
