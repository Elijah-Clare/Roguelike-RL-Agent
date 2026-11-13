# PROGRAMMER: Dashawn Williams
# DUE DATE: Monday, 11/30/26
# Tests for enemies, combat, and dying.

from roguelike.dungeon import WALL, distances_from
from roguelike.enemies import GOBLIN, RAT, Enemy, chase_steps
from roguelike.env import ENEMY_CH, VIEW_SIZE, RoguelikeEnv
from roguelike.game import MIN_ENEMY_DISTANCE, MOVES, PLAYER_DAMAGE, WAIT, Game


def open_neighbor(game):
    """an action and the open tile next to the player it leads to"""
    for action, (d_row, d_col) in MOVES.items():
        tile = (game.pos[0] + d_row, game.pos[1] + d_col)
        if game.grid[tile] != WALL and tile != game.dungeon.stairs:
            return action, tile
    raise AssertionError("player is boxed in")


def test_enemies_spawn_away_from_start():
    """1 to 3 enemies spawn on open tiles far enough from the start"""
    for seed in range(100):
        game = Game(seed)
        dist = distances_from(game.grid, game.pos)
        assert 1 <= len(game.enemies) <= 3
        positions = [enemy.pos for enemy in game.enemies]
        assert len(set(positions)) == len(positions)
        for enemy in game.enemies:
            assert dist[enemy.pos] >= MIN_ENEMY_DISTANCE
            assert enemy.pos != game.dungeon.stairs


def test_bump_attacks_and_kills():
    """moving into an enemy hurts it, and enough hits kill it"""
    game = Game(seed=2)
    action, tile = open_neighbor(game)
    goblin = Enemy(GOBLIN, tile, GOBLIN.max_hp)
    game.enemies = [goblin]
    start = game.pos

    events = game.step(action)
    assert events.attacked and not events.moved and game.pos == start
    assert goblin.hp == GOBLIN.max_hp - PLAYER_DAMAGE

    events = game.step(action)
    assert events.killed_enemy and game.enemies == []


def test_adjacent_enemy_hits_player():
    """an enemy next to the player attacks on its turn"""
    game = Game(seed=2)
    _, tile = open_neighbor(game)
    game.enemies = [Enemy(GOBLIN, tile, 100)]
    events = game.step(WAIT)
    assert events.damage_taken == GOBLIN.damage
    assert game.hp == game.max_hp - GOBLIN.damage


def test_player_can_die():
    """losing all hp ends the run as a death"""
    game = Game(seed=2)
    _, tile = open_neighbor(game)
    game.enemies = [Enemy(GOBLIN, tile, 100)]
    game.hp = 1
    events = game.step(WAIT)
    assert events.died and game.died and game.done and game.hp == 0


def test_chase_moves_closer():
    """chase steps always shrink the distance to the target"""
    assert chase_steps((5, 5), (5, 8)) == [(0, 1)]
    assert chase_steps((5, 5), (2, 4)) == [(-1, 0), (0, -1)]
    assert chase_steps((5, 5), (6, 9)) == [(0, 1), (1, 0)]


def test_enemies_stay_on_open_tiles():
    """enemies never walk into walls, the player, or each other"""
    game = Game(seed=8)
    for _ in range(150):
        if game.done:
            break
        game.step(WAIT)
        positions = [enemy.pos for enemy in game.enemies]
        assert len(set(positions)) == len(positions)
        for pos in positions:
            assert game.grid[pos] != WALL and pos != game.pos


def test_died_is_terminated_in_env():
    """dying ends the episode as terminated"""
    env = RoguelikeEnv()
    env.reset(seed=2)
    _, tile = open_neighbor(env.game)
    env.game.enemies = [Enemy(RAT, tile, 100)]
    env.game.hp = 1
    _, _, terminated, truncated, info = env.step(WAIT)
    assert terminated and not truncated and info["events"]["died"]


def test_enemy_shows_in_view():
    """enemies appear in the enemy channel at the right spot"""
    env = RoguelikeEnv()
    env.reset(seed=2)
    _, tile = open_neighbor(env.game)
    env.game.enemies = [Enemy(RAT, tile, RAT.max_hp)]
    obs = env.observe()
    half = VIEW_SIZE // 2
    row, col = tile[0] - env.game.pos[0] + half, tile[1] - env.game.pos[1] + half
    assert obs["view"][ENEMY_CH, row, col] == RAT.obs_value
    assert obs["view"][ENEMY_CH].sum() == RAT.obs_value
