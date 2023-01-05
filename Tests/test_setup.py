## Functions to test
##  map.next_cell_to_move(origin, dest)
##  map.steps_to_location(origin, dest)
##  map.gps_to_regions_grid(gps)
##  map.gps_to_regions_hexagon(gps)

import Utils.setup as setup

def test_graph():
    ## Setup a graph system
    graph_edge_lst = [
        (0, 0), (0, 1), (0, 2),
        (1, 1), (1, 2),
        (2, 2), (2, 0), (2, 3),
        (3, 3), (3, 0), (3, 4),
        (4, 4), (4, 3)
    ]
    map = setup.Map(map_system = "graph", num_nodes = 5, graph_edge_lst = graph_edge_lst)
    ## Test next_cell_to_move
    assert map.next_cell_to_move(0, 0) == 0
    assert map.next_cell_to_move(0, 1) == 1
    assert map.next_cell_to_move(0, 2) == 2
    assert map.next_cell_to_move(0, 3) == 2
    assert map.next_cell_to_move(0, 4) == 2
    assert map.next_cell_to_move(1, 0) == 2
    assert map.next_cell_to_move(1, 1) == 1
    assert map.next_cell_to_move(1, 2) == 2
    assert map.next_cell_to_move(1, 3) == 2
    assert map.next_cell_to_move(1, 4) == 2
    assert map.next_cell_to_move(2, 0) == 0
    assert map.next_cell_to_move(2, 1) == 0
    assert map.next_cell_to_move(2, 2) == 2
    assert map.next_cell_to_move(2, 3) == 3
    assert map.next_cell_to_move(2, 4) == 3
    assert map.next_cell_to_move(3, 0) == 0
    assert map.next_cell_to_move(3, 1) == 0
    assert map.next_cell_to_move(3, 2) == 0
    assert map.next_cell_to_move(3, 3) == 3
    assert map.next_cell_to_move(3, 4) == 4
    assert map.next_cell_to_move(4, 0) == 3
    assert map.next_cell_to_move(4, 1) == 3
    assert map.next_cell_to_move(4, 2) == 3
    assert map.next_cell_to_move(4, 3) == 3
    assert map.next_cell_to_move(4, 4) == 4
    
    ## Test steps_to_location
    assert map.steps_to_location(0, 0) == 0
    assert map.steps_to_location(0, 1) == 1
    assert map.steps_to_location(0, 2) == 1
    assert map.steps_to_location(0, 3) == 2
    assert map.steps_to_location(0, 4) == 3
    assert map.steps_to_location(1, 0) == 2
    assert map.steps_to_location(1, 1) == 0
    assert map.steps_to_location(1, 2) == 1
    assert map.steps_to_location(1, 3) == 2
    assert map.steps_to_location(1, 4) == 3
    assert map.steps_to_location(2, 0) == 1
    assert map.steps_to_location(2, 1) == 2
    assert map.steps_to_location(2, 2) == 0
    assert map.steps_to_location(2, 3) == 1
    assert map.steps_to_location(2, 4) == 2
    assert map.steps_to_location(3, 0) == 1
    assert map.steps_to_location(3, 1) == 2
    assert map.steps_to_location(3, 2) == 2
    assert map.steps_to_location(3, 3) == 0
    assert map.steps_to_location(3, 4) == 1
    assert map.steps_to_location(4, 0) == 2
    assert map.steps_to_location(4, 1) == 3
    assert map.steps_to_location(4, 2) == 3
    assert map.steps_to_location(4, 3) == 1
    assert map.steps_to_location(4, 4) == 0

def test_grid():
    ## Setup a grid system
    map = setup.Map(map_system = "grid", num_layers = 2, lon_range = (40, 70), lat_range = (50, 80))
    ## Test next_cell_to_move
    assert map.next_cell_to_move(0, 0) == 0
    assert map.next_cell_to_move(3, 8) in [4, 6]
    assert map.next_cell_to_move(2, 6) in [1, 5]
    ## Test steps_to_location
    assert map.steps_to_location(0, 4) == 2
    assert map.steps_to_location(3, 3) == 0
    assert map.steps_to_location(0, 8) == 4
    assert map.steps_to_location(5, 6) == 3
    assert map.steps_to_location(7, 4) == 1
    ## Test gps_to_regions_grid
    assert map.gps_to_regions_grid((41, 52)) == 0
    assert map.gps_to_regions_grid((49.9, 80)) == 6
    assert map.gps_to_regions_grid((65, 61)) == 5
    assert map.gps_to_regions_grid((73, 84)) == 8

def test_hexagon():
    ## Setup a hexagon system
    map = setup.Map(map_system = "hexagon", num_layers = 3, lon_range = (30, 60), lat_range = (40, 65))
    id_ = lambda x: map.hexagon_triple_to_id[x]
    triple_ = lambda x: map.hexagon_id_to_triple[x]
    ## Test next_cell_to_move
    assert map.next_cell_to_move(id_((0, 0, 0)), id_((0, 0, 0))) == id_((0, 0, 0))
    assert map.next_cell_to_move(id_((0, 0, 0)), id_((-1, 2, -1))) in [id_((-1, 1, 0)), id_((0, 1, -1))]
    assert map.next_cell_to_move(id_((2, -2, 0)), id_((-1, 2, -1))) in [id_((2, -1, -1)), id_((1, -1, 0))]
    ## Test steps_to_location
    assert map.steps_to_location(id_((0, 0, 0)), id_((0, 0, 0))) == 0
    assert map.steps_to_location(id_((0, 0, 0)), id_((-1, 2, -1))) == 2
    assert map.steps_to_location(id_((2, -2, 0)), id_((-1, 2, -1))) == 4
    ## Test gps_to_regions_grid
    assert map.gps_to_regions_hexagon((46, 54)) == id_((0, 0, 0))
    assert map.gps_to_regions_hexagon((30, 60)) == id_((2, 0, -2))
    assert map.gps_to_regions_hexagon((45, 66)) == id_((0, 2, -2))
    assert map.gps_to_regions_hexagon((57.5, 60)) == id_((-2, 2, 0))
    assert map.gps_to_regions_hexagon((52.5, 64)) == id_((-1, 2, -1))
