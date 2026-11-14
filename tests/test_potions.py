# PROGRAMMER: Dashawn Williams
# DUE DATE: Monday, 11/30/26
# Tests for picking up and drinking potions.

from roguelike.dungeon import FLOOR, WALL
from roguelike.env import POTION_CH, VIEW_SIZE, RoguelikeEnv
from roguelike.game import DRINK, MAX_POTIONS, MOVES, POTION_HEAL, Game


def quiet_game(seed=2):
    """game with no enemies or potions on the map"""
    game = Game(seed)
    game.enemies = []
    game.potion_tiles = set()
    return game


def open_neighbor(game):
    """an action and the open floor tile next to the player it leads to"""
    for action, (d_row, d_col) in MOVES.items():
        tile = (game.pos[0] + d_row, game.pos[1] + d_col)
        if game.grid[tile] == FLOOR:
            return action, tile
    raise AssertionError("no open floor next to the player")


def test_potions_spawn_on_floor():
    """1 or 2 potions spawn on floor tiles away from the player and enemies"""
    for seed in range(100):
        game = Game(seed)
        assert 1 <= len(game.potion_tiles) <= 2
        enemy_tiles = {enemy.pos for enemy in game.enemies}
        for tile in game.potion_tiles:
            assert game.grid[tile] == FLOOR
            assert tile != game.pos and tile not in enemy_tiles


def test_walking_on_potion_picks_it_up():
    """stepping on a potion adds it to the player's potions"""
    game = quiet_game()
    action, tile = open_neighbor(game)
    game.potion_tiles = {tile}
    events = game.step(action)
    assert events.picked_potion and game.potions == 1 and not game.potion_tiles


def test_cannot_carry_more_than_max():
    """a full player leaves the potion on the floor"""
    game = quiet_game()
    action, tile = open_neighbor(game)
    game.potion_tiles = {tile}
    game.potions = MAX_POTIONS
    events = game.step(action)
    assert not events.picked_potion and game.potions == MAX_POTIONS and tile in game.potion_tiles


def test_drinking_heals():
    """drinking uses a potion and restores hp"""
    game = quiet_game()
    game.potions, game.hp = 1, 2
    events = game.step(DRINK)
    assert events.drank_potion and events.healed == POTION_HEAL
    assert game.hp == 2 + POTION_HEAL and game.potions == 0


def test_healing_stops_at_max_hp():
    """healing never goes above max hp"""
    game = quiet_game()
    game.potions, game.hp = 1, game.max_hp - 1
    events = game.step(DRINK)
    assert events.healed == 1 and game.hp == game.max_hp


def test_drinking_with_none_does_nothing():
    """drink with no potions changes nothing"""
    game = quiet_game()
    game.hp = 4
    events = game.step(DRINK)
    assert not events.drank_potion and game.hp == 4


def test_potions_show_in_view_and_stats():
    """potions on the map and potions carried both reach the agent"""
    env = RoguelikeEnv()
    env.reset(seed=2)
    game = env.game
    game.enemies = []
    _, tile = open_neighbor(game)
    game.potion_tiles = {tile}
    game.potions = 1
    obs = env.observe()
    half = VIEW_SIZE // 2
    assert obs["view"][POTION_CH, tile[0] - game.pos[0] + half, tile[1] - game.pos[1] + half] == 1.0
    assert obs["view"][POTION_CH].sum() == 1.0
    assert obs["stats"][3] == 1 / MAX_POTIONS
