# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Joshua Levine (joshua45@illinois.edu)
# Inspired by work done by James Gao (jamesjg2@illinois.edu) and Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP5
"""

import numpy as np
from alien import Alien
from typing import List, Tuple
from copy import deepcopy


def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endy), ...]

        Return:
            True if touched, False if not
    """
    head, tail = alien.get_head_and_tail()
    pos = alien.get_centroid()
    if(head[0] - tail[0] == 0 and head[1] - tail[1] == 0):
        for wall in walls:
           if point_segment_distance(pos,((wall[0], wall[1]), (wall[2], wall[3]))) <= alien.get_width():
               return True
        return False

    elif(head[0] - tail[0] == 0):
        x_buffer = alien.get_width()
        y_buffer = alien.get_length()/2.0
    elif(head[1] - tail[1] == 0):
        x_buffer = alien.get_length()/2.0
        y_buffer = alien.get_width() 

    for wall in walls:
        if point_segment_distance(head,((wall[0], wall[1]), (wall[2], wall[3]))) <= alien.get_width():
            return True
        if point_segment_distance(tail,((wall[0], wall[1]), (wall[2], wall[3]))) <= alien.get_width():
            return True
        # if point_segment_distance(alien.get_centroid(),((wall[0], wall[1]), (wall[2], wall[3]))) > alien.get_length()/2.0 + alien.get_width():
        #     continue
        
        angle = np.degrees(np.arctan2(wall[3] - wall[1], wall[2] - wall[0])) % 360

        if angle == 0:
            buffer = ((wall[0] - x_buffer, wall[1] + y_buffer),(wall[0] - x_buffer, wall[1] - y_buffer),
                      (wall[2] + x_buffer, wall[3] - y_buffer),(wall[2] + x_buffer, wall[3] + y_buffer))
        elif angle == 90:
            buffer = ((wall[0] - x_buffer, wall[1] - y_buffer),(wall[0] + x_buffer, wall[1] - y_buffer),
                      (wall[2] + x_buffer, wall[3] + y_buffer),(wall[2] - x_buffer, wall[3] + y_buffer))
        elif angle == 180:
            buffer = ((wall[0] + x_buffer, wall[1] + y_buffer),(wall[0] + x_buffer, wall[1] - y_buffer),
                      (wall[2] - x_buffer, wall[3] - y_buffer),(wall[2] - x_buffer, wall[3] + y_buffer))
        elif angle == 270:
            buffer = ((wall[0] - x_buffer, wall[1] + y_buffer),(wall[0] + x_buffer, wall[1] + y_buffer),
                      (wall[2] + x_buffer, wall[3] - y_buffer),(wall[2] - x_buffer, wall[3] - y_buffer))
        elif angle > 270:
            buffer = ((wall[0] + x_buffer, wall[1] + y_buffer),(wall[0] - x_buffer, wall[1] + y_buffer),(wall[0] - x_buffer, wall[1] - y_buffer), 
                      (wall[2] - x_buffer, wall[3] - y_buffer),(wall[2] + x_buffer, wall[3] - y_buffer),(wall[2] + x_buffer, wall[3] + y_buffer),)
        elif angle > 180:
            buffer = ((wall[0] - x_buffer, wall[1] + y_buffer),(wall[0] + x_buffer, wall[1] + y_buffer),(wall[0] + x_buffer, wall[1] - y_buffer),
                      (wall[2] + x_buffer, wall[3] - y_buffer),(wall[2] - x_buffer, wall[3] - y_buffer),(wall[2] - x_buffer, wall[3] + y_buffer))
        elif angle > 90:
            buffer = ((wall[0] - x_buffer, wall[1] - y_buffer),(wall[0] + x_buffer, wall[1] - y_buffer),(wall[0] + x_buffer, wall[1] + y_buffer),
                      (wall[2] + x_buffer, wall[3] + y_buffer),(wall[2] - x_buffer, wall[3] + y_buffer),(wall[2] - x_buffer, wall[3] - y_buffer),)
        else:
            buffer = ((wall[0] - x_buffer, wall[1] + y_buffer),(wall[0] - x_buffer, wall[1] - y_buffer),(wall[0] + x_buffer, wall[1] - y_buffer),
                      (wall[2] + x_buffer, wall[3] - y_buffer),(wall[2] + x_buffer, wall[3] + y_buffer),(wall[2] - x_buffer, wall[3] + y_buffer))

        if is_point_in_polygon(alien.get_centroid(), buffer):
            return True

    return False


def is_alien_within_window(alien: Alien, window: Tuple[int]):
    """Determine whether the alien stays within the window

        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
    """
    head, tail = alien.get_head_and_tail()
    pos = alien.get_centroid()
    if(head[0] - tail[0] == 0 and head[1] - tail[1] == 0):
        x_buffer = y_buffer = alien.get_width()
    elif(head[0] - tail[0] == 0):
        x_buffer = alien.get_width()
        y_buffer = alien.get_length()/2.0 + alien.get_width()
    elif(head[1] - tail[1] == 0):
        x_buffer = alien.get_length()/2.0 + alien.get_width()
        y_buffer = alien.get_width() 

    return pos[0] > x_buffer and pos[0] < window[0] - x_buffer and pos[1] > y_buffer and pos[1] < window[1] - y_buffer


def is_point_in_polygon(point, polygon):
    """Determine whether a point is in a parallelogram.
    Note: The vertex of the parallelogram should be clockwise or counter-clockwise.

        Args:
            point (tuple): shape of (2, ). The coordinate (x, y) of the query point.
            polygon (tuple): shape of (4, 2). The coordinate (x, y) of 4 vertices of the parallelogram.
    """
    sum = 0
    inf_y = max(polygon, key=lambda x:x[1])[1]
    if inf_y < point[1]:
        return False
    inf_point = (point[0], inf_y + 1)
    
    for i in range(-1, len(polygon)-1):
        if point_segment_distance(point, (polygon[i],polygon[i+1])) == 0:
            return True
        if do_segments_intersect((point, inf_point), (polygon[i],polygon[i+1]), improper_intersection="half"):
            sum+=1
    
    return sum % 2 == 1



def does_alien_path_touch_wall(alien: Alien, walls: List[Tuple[int]], waypoint: Tuple[int, int]):
    """Determine whether the alien's straight-line path from its current position to the waypoint touches a wall

        Args:
            alien (Alien): the current alien instance
            walls (List of tuple): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]
            waypoint (tuple): the coordinate of the waypoint where the alien wants to move

        Return:
            True if touched, False if not
    """

    head, tail = alien.get_head_and_tail()
    pos = alien.get_centroid()

    if does_alien_touch_wall(alien, walls):
        return True

    if(head[0] - tail[0] == 0 and head[1] - tail[1] == 0):
        for wall in walls:
           if segment_distance((pos,waypoint),((wall[0], wall[1]), (wall[2], wall[3]))) <= alien.get_width():
               return True
        return False

    elif(head[0] - tail[0] == 0):
        x_buffer = alien.get_width()
        y_buffer = alien.get_length()/2.0

    elif(head[1] - tail[1] == 0):
        x_buffer = alien.get_length()/2.0
        y_buffer = alien.get_width()

    vertexes = set()
    for wall in walls:
        if segment_distance((head, (waypoint[0] + (head[0] - pos[0]), waypoint[1] + (head[1] - pos[1]))),((wall[0], wall[1]), (wall[2], wall[3]))) <= alien.get_width():
            return True
        if segment_distance((tail, (waypoint[0] + (tail[0] - pos[0]), waypoint[1] + (tail[1] - pos[1]))),((wall[0], wall[1]), (wall[2], wall[3]))) <= alien.get_width():
            return True
        
        vertexes.add((wall[0], wall[1]))
        vertexes.add((wall[2], wall[3]))
    
    angle = np.degrees(np.arctan2(waypoint[1] - pos[1], waypoint[0] - pos[0])) % 360
    if angle == 0:
        buffer = ((pos[0] - x_buffer, pos[1] + y_buffer),(pos[0] - x_buffer, pos[1] - y_buffer),
                  (waypoint[0] + x_buffer, waypoint[1] - y_buffer),(waypoint[0] + x_buffer, waypoint[1] + y_buffer))
    elif angle == 90:
        buffer = ((pos[0] - x_buffer, pos[1] - y_buffer),(pos[0] + x_buffer, pos[1] - y_buffer),
                  (waypoint[0] + x_buffer, waypoint[1] + y_buffer),(waypoint[0] - x_buffer, waypoint[1] + y_buffer))
    elif angle == 180:
        buffer = ((pos[0] + x_buffer, pos[1] + y_buffer),(pos[0] + x_buffer, pos[1] - y_buffer),
                  (waypoint[0] - x_buffer, waypoint[1] - y_buffer),(waypoint[0] - x_buffer, waypoint[1] + y_buffer))
    elif angle == 270:
        buffer = ((pos[0] - x_buffer, pos[1] + y_buffer),(pos[0] + x_buffer, pos[1] + y_buffer),
                  (waypoint[0] + x_buffer, waypoint[1] - y_buffer),(waypoint[0] - x_buffer, waypoint[1] - y_buffer))
    elif angle > 270:
        buffer = ((pos[0] + x_buffer, pos[1] + y_buffer),(pos[0] - x_buffer, pos[1] + y_buffer),(pos[0] - x_buffer, pos[1] - y_buffer), 
                  (waypoint[0] - x_buffer, waypoint[1] - y_buffer),(waypoint[0] + x_buffer, waypoint[1] - y_buffer),(waypoint[0] + x_buffer, waypoint[1] + y_buffer),)
    elif angle > 180:
        buffer = ((pos[0] - x_buffer, pos[1] + y_buffer),(pos[0] + x_buffer, pos[1] + y_buffer),(pos[0] + x_buffer, pos[1] - y_buffer),
                  (waypoint[0] + x_buffer, waypoint[1] - y_buffer),(waypoint[0] - x_buffer, waypoint[1] - y_buffer),(waypoint[0] - x_buffer, waypoint[1] + y_buffer))
    elif angle > 90:
        buffer = ((pos[0] - x_buffer, pos[1] - y_buffer),(pos[0] + x_buffer, pos[1] - y_buffer), (pos[0] + x_buffer, pos[1] + y_buffer),
                  (waypoint[0] + x_buffer, waypoint[1] + y_buffer),(waypoint[0] - x_buffer, waypoint[1] + y_buffer), (waypoint[0] - x_buffer, waypoint[1] - y_buffer))
    else:
        buffer = ((pos[0] - x_buffer, pos[1] + y_buffer),(pos[0] - x_buffer, pos[1] - y_buffer), (pos[0] + x_buffer, pos[1] - y_buffer),
                  (waypoint[0] + x_buffer, waypoint[1] - y_buffer),(waypoint[0] + x_buffer, waypoint[1] + y_buffer), (waypoint[0] - x_buffer, waypoint[1] + y_buffer))

    for vertex in vertexes:
        if is_point_in_polygon(vertex, buffer):
            return True

    return False


def point_segment_distance(p, s):
    """Compute the distance from the point to the line segment.

        Args:
            p: A tuple (x, y) of the coordinates of the point.
            s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.
    """
    line_vec = np.array([(s[1][0] - s[0][0]), (s[1][1] - s[0][1])])
    line_len = (line_vec[0]**2 + line_vec[1]**2)**0.5
    point_vec = np.array([(p[0] - s[0][0] ),(p[1] - s[0][1] )])

    if line_len == 0:
        nearest_point = s[0]
    else:
        scalar_proj = np.dot(line_vec,point_vec)/(line_len**2)
        scalar_proj_norm = max(0, min(scalar_proj, 1))
        nearest_point = ((scalar_proj_norm) * line_vec) + s[0]

    dist = ((p[0] - nearest_point[0])**2 + (p[1] - nearest_point[1])**2)**0.5
    
    if dist < 10**-8:
        dist = 0
    return dist


def do_segments_intersect(s1, s2, improper_intersection = "full"):
    """Determine whether segment1 intersects segment2.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """
    if improper_intersection == "false":
        case = 0
    elif improper_intersection == "half":
        case = 1
    else: 
        case = 2


    A,B = s1
    C,D = s2

    oA = np.sign((D[1] - C[1]) * (A[0] - D[0]) - (D[0] - C[0]) * (A[1] - D[1]))
    oB = np.sign((D[1] - C[1]) * (B[0] - D[0]) - (D[0] - C[0]) * (B[1] - D[1]))
    oC = np.sign((B[1] - A[1]) * (C[0] - B[0]) - (B[0] - A[0]) * (C[1] - B[1]))
    oD = np.sign((B[1] - A[1]) * (D[0] - B[0]) - (B[0] - A[0]) * (D[1] - B[1]))


    if oA != oB and oC != oD and oA !=0 and oB != 0 and oC != 0 and oD != 00:
        return True

    if oA == 0 and point_segment_distance(A,(C,D)) == 0 and case != 0:
        return True
    if oB == 0 and point_segment_distance(B,(C,D)) == 0 and case == 2:
        return True
    if oC == 0 and point_segment_distance(C,(A,B)) == 0 and case != 0:
        return True
    if oD == 0 and point_segment_distance(D,(A,B)) == 0 and case == 2:
        return True
    
    return False


def segment_distance(s1, s2):
    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """
    if do_segments_intersect(s1,s2):
        return 0

    d1 = point_segment_distance(s1[0], s2)
    d2 = point_segment_distance(s1[1], s2)
    d3 = point_segment_distance(s2[0], s1)
    d4 = point_segment_distance(s2[1], s1)

    return min(d1,d2,d3,d4)


if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result, waypoints


    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                        f'{b} is expected to be {result[i]}, but your' \
                                                                        f'result is {distance}'


    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls)
        in_window_result = is_alien_within_window(alien, window)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'


    def test_check_path(alien: Alien, position, truths, waypoints):
        alien.set_alien_pos(position)
        config = alien.get_config()

        for i, waypoint in enumerate(waypoints):
            path_touch_wall_result = does_alien_path_touch_wall(alien, walls, waypoint)

            assert path_touch_wall_result == truths[
                i], f'does_alien_path_touch_wall(alien, walls, waypoint) with alien config {config} ' \
                    f'and waypoint {waypoint} returns {path_touch_wall_result}, ' \
                    f'expected: {truths[i]}'

            # Initialize Aliens and perform simple sanity check.
    

    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)

    # Test validity of straight line paths between an alien and a waypoint
    test_check_path(alien_ball, (30, 120), (False, True, True), waypoints)
    test_check_path(alien_horz, (30, 120), (False, True, False), waypoints)
    test_check_path(alien_vert, (30, 120), (True, True, True), waypoints)

    centers = alien_positions
    segments = walls
    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")
