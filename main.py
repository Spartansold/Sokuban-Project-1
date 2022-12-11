import collections
import heapq
import time

import numpy as np

# ###################################################################################################
# makeGameState
# Input: the starting map
# Output: the map converted into all possible states, 0= empty, 1 = wall, 2 = robot, 3= box,
# 4 = goal, 5= box and goal, 6= robot and goal
# ###################################################################################################
def game_state(map):
    height = len(map)
    map = [x.replace('\n', '') for x in map]
    map = [','.join(map[i]) for i in range(height)]
    map = [x.split(',') for x in map]
    maxWeight = max([len(x) for x in map])

    save_walls = []

    for i in range(height):
        for j in range(len(map[i])):
            if map[i][j] == ' ':
                map[i][j] = 0
            elif map[i][j] == 'O':
                map[i][j] = 1  # wall
                save_walls += [(i, j)]
            elif map[i][j] == 'R':
                map[i][j] = 2  # robot
            elif map[i][j] == 'B':
                map[i][j] = 3  # box
            elif map[i][j] == 'S':
                map[i][j] = 4  # goal
            elif map[i][j] == 'SB':
                map[i][j] = 5  # box and goal
            elif map[i][j] == 'SR':
                map[i][j] = 6  # robot and goal
        colsNum = len(map[i])
        if colsNum < maxWeight:
            map[i].extend([1 for _ in range(maxWeight - colsNum)])

    array = np.array(map)
    posWalls = tuple(tuple(x) for x in np.argwhere(array == 1))
    posGoals = tuple(tuple(x) for x in np.argwhere((array == 4) | (array == 5) | (array == 6)))
    initialBoxs = tuple(tuple(x) for x in np.argwhere((array == 3) | (array == 5)))
    initialActor = tuple(np.argwhere((array == 2) | (array == 6))[0])

    return posWalls, posGoals, initialBoxs, initialActor, save_walls


####################################################################################################
# check_if_won
# Input: Box locations
# Output: True if no more boxes, False if there are boxes left
####################################################################################################
def check_if_won(posBoxs):
    for box in posBoxs:
        if box not in posGoals:
            return False
    return True


####################################################################################################
# is valid move
# Input: the move that is being made, the robot locations, box locations
# Output: True if valid, False if not valid
####################################################################################################
def is_valid_move(move, posActor, posBoxs):
    xActor, yActor = posActor
    if move[-1]:
        xNext, yNext = xActor + 2 * move[0], yActor + 2 * move[1]
    else:
        xNext, yNext = xActor + move[0], yActor + move[1]
    return (xNext, yNext) not in posBoxs + posWalls


####################################################################################################
# next_moves
# Input: where the robot and boxes are
# Output: all valid movements
####################################################################################################
def next_moves(posActor, posBoxs):
    xActor, yActor = posActor
    allNextMoves = []
    for move in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
        xNext, yNext = xActor + move[0], yActor + move[1]
        if (xNext, yNext) in posBoxs:
            move.append(True)
        else:
            move.append(False)  #
        if is_valid_move(move, posActor, posBoxs):
            allNextMoves.append(move)
    return tuple(tuple(x) for x in allNextMoves)


####################################################################################################
# update_state
# Input: robot, boxes, and move
# Output: the new position of the robot and boxes
# moves the robot and boxes to the next location
####################################################################################################
def update_state(posRobot, posBoxs, move):
    xActor, yActor = posRobot
    newPosActor = [xActor + move[0], yActor + move[1]]
    posBoxs = [list(x) for x in posBoxs]

    if move[-1]:
        posBoxs.remove(newPosActor)
        posBoxs.append([xActor + 2 * move[0], yActor + 2 * move[1]])

    newPosBoxs = tuple(tuple(x) for x in posBoxs)
    newPosActor = tuple(newPosActor)
    return newPosActor, newPosBoxs


####################################################################################################
# check_failed
# Input: box location
# Output: True if the box is stuck, False if the box is not stuck
# if a box is next to two walls as in a corner and not in a goal space, then it is stuck
####################################################################################################
def check_failed(posBoxs):
    for box in posBoxs:
        if box in posGoals:
            continue

        if (box[0] + 1, box[1]) in posWalls:
            if ((box[0], box[1] + 1) in posWalls) or (
                    (box[0], box[1] - 1) in posWalls):
                return True
            if ((box[0], box[1] + 1) in posBoxs) and (
                    box[0] + 1, box[1] + 1) in posWalls:
                return True
            if ((box[0], box[1] - 1) in posBoxs) and (
                    box[0] + 1, box[1] - 1) in posWalls:
                return True

        if (box[0] - 1, box[1]) in posWalls:
            if ((box[0], box[1] + 1) in posWalls) or (
                    (box[0], box[1] - 1) in posWalls):
                return True
            if ((box[0], box[1] + 1) in posBoxs) and (
                    box[0] - 1, box[1] + 1) in posWalls:
                return True
            if ((box[0], box[1] - 1) in posBoxs) and (
                    box[0] - 1, box[1] - 1) in posWalls:
                return True

        if (box[0], box[1] + 1) in posWalls:
            if ((box[0] + 1, box[1]) in posBoxs) and (
                    box[0] + 1, box[1] + 1) in posWalls:
                return True
            if ((box[0] - 1, box[1]) in posBoxs) and (
                    box[0] - 1, box[1] + 1) in posWalls:
                return True

        if (box[0], box[1] - 1) in posWalls:
            if ((box[0] + 1, box[1]) in posBoxs) and (
                    box[0] + 1, box[1] - 1) in posWalls:
                return True
            if ((box[0] - 1, box[1]) in posBoxs) and (
                    box[0] - 1, box[1] - 1) in posWalls:
                return True
    return False


####################################################################################################
# heuristicFunction
# Input: the box location
# Output: Manhattan distance to goal space
# determines the heuristic by calculating the manhattan distance of each box to the goal
####################################################################################################
def heuristic_function(posBoxs):
    distance = 0
    completes = set(posGoals) & set(posBoxs)
    sortedPosBoxs = list(set(posBoxs).difference(completes))
    sortedPosGoals = list(set(posGoals).difference(completes))

    for i in range(len(sortedPosBoxs)):
        distance += (abs(sortedPosBoxs[i][0] - sortedPosGoals[i][0])) + (
            abs(sortedPosBoxs[i][1] - sortedPosGoals[i][1]))
    return distance


####################################################################################################
# costFunction
# Input: the node
# Output: the len to the next node
####################################################################################################
def cost_function(node):
    return len(node)


####################################################################################################
# aStarAlgorithm
# Input: 
# Output:
# Uses both heuristic and the length to determine which direction is the best direction
####################################################################################################
def aStarAlgorithm():
    priorityQueue = []
    initialNode = [(initialActor, initialBoxs)]
    heapq.heappush(priorityQueue, (0, initialNode))
    exploredSet = set()

    while priorityQueue:
        (_, node) = heapq.heappop(priorityQueue)
        if check_if_won(node[-1][-1]):
            return node
        if node[-1] in exploredSet:
            continue
        exploredSet.add(node[-1])
        cost = cost_function(node)
        allNextMoves = next_moves(node[-1][0], node[-1][1])
        for move in allNextMoves:
            newPosActor, newPosBox = update_state(node[-1][0], node[-1][1], move)
            if not check_failed(newPosBox):
                saveNode = node + [(newPosActor, newPosBox)]
                priority = heuristic_function(newPosBox) + cost
                heapq.heappush(priorityQueue, (priority, saveNode))
    return []


####################################################################################################
# greedy_algorithm
# Input:
# Output: the path to the goal
# Uses the heuristic to determine which direction it should go
####################################################################################################
def greedy_algorithm():
    priorityQueue = []
    initialNode = [(initialActor, initialBoxs)]
    heapq.heappush(priorityQueue, (0, initialNode))
    exploredSet = set()

    while priorityQueue:
        (_, node) = heapq.heappop(priorityQueue)
        if check_if_won(node[-1][-1]):
            return node
        if node[-1] in exploredSet:
            continue
        exploredSet.add(node[-1])
        allNextMoves = next_moves(node[-1][0], node[-1][1])
        for move in allNextMoves:
            newPosActor, newPosBox = update_state(node[-1][0], node[-1][1], move)
            if not check_failed(newPosBox):
                saveNode = node + [(newPosActor, newPosBox)]
                priority = heuristic_function(newPosBox)
                heapq.heappush(priorityQueue, (priority, saveNode))
    return []


####################################################################################################
# DFS_algorithm
# Input: 
# Output: the path to the goal
# Goes through movements using a fifo stack
####################################################################################################
def DFS_algorithm():
    stack = collections.deque([[(initialActor, initialBoxs)]])
    exploredSet = set()

    while stack:
        node = stack.pop()
        if check_if_won(node[-1][-1]):
            return node
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            for action in next_moves(node[-1][0], node[-1][1]):
                newPosPlayer, newPosBox = update_state(node[-1][0], node[-1][1], action)
                if not check_failed(newPosBox):
                    stack.append(node + [(newPosPlayer, newPosBox)])
    return []


####################################################################################################
# BFS_algorithm
# Input:
# Output: the path to the goal
# Goes through all possible moves using a lifo heap
####################################################################################################
def BFS_algorithm():
    heap = []
    initialNode = [(initialActor, initialBoxs)]
    heapq.heappush(heap, initialNode)
    exploredSet = set()

    while heap:
        node = heapq.heappop(heap)
        if check_if_won(node[-1][-1]):
            return node
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            for action in next_moves(node[-1][0], node[-1][1]):
                newPosPlayer, newPosBox = update_state(node[-1][0], node[-1][1], action)
                if not check_failed(newPosBox):
                    heap.append(node + [(newPosPlayer, newPosBox)])
    return []


####################################################################################################
# Print result of the game, save it in outputs folder
####################################################################################################
def print_result(alg, result):
    maxHeight = len(initial)
    maxWidth = max([len(i) for i in initial])
    with open("outputs/"+alg + "_" + filename, "w") as f:
        for rs in result:
            for i in range(maxHeight):
                for j in range(maxWidth - 1):
                    ch = ' '
                    position = (i, j)
                    if position in posGoals:
                        if position in rs[1]:
                            ch = 'SB'
                        elif position == rs[0]:
                            ch = 'SR'
                        else:
                            ch = 'S'
                    elif position in saveWalls:
                        ch = 'O'
                    elif position in rs[1]:
                        ch = 'B'
                    elif position == rs[0]:
                        ch = 'R'
                    f.write(ch)
                f.write('\n')
            f.write('\n')


####################################################################################################
# MAIN
####################################################################################################
if __name__ == '__main__':
    filename = input("Type the file name: ")
    with open("inputs/" + filename, "r") as f:
        initial = f.readlines()
    while True:
        alg = input("Select search algorithm (1 - DFS algorithm, 2 - BFS algorithm, 3 - greedy algorithm, 4- A star "
                    "algorithm, 5 - all algorithms): ")
        if alg in ["1", "2", "3", "4", "5"]:
            break

    posWalls, posGoals, initialBoxs, initialActor, saveWalls = game_state(initial)

    startTime = time.time()
    if alg == "1":
        print("Using the DFS algorithm to solve...")
        result = DFS_algorithm()
        endTime = time.time()

        if result:
            print_result("DFS", result)
            print("Completeness: True")
            print("Runtime: {0} second.".format(endTime - startTime))
            print("Total step: ", len(result))
        else:
            print("Completeness: False")
            print("Runtime: {0} second.".format(endTime - startTime))
            print("Total step: ", len(result))

    elif alg == "2":
        print("Using BFS algorithm to solve...")
        result = BFS_algorithm()
        endTime = time.time()

        if result:
            print_result("BFS", result)
            print("Completeness: True")
            print("Runtime: {0} second.".format(endTime - startTime))
            print("Total step: ", len(result))
        else:
            print("Completeness: False")
            print("Runtime: {0} second.".format(endTime - startTime))
            print("Total step: ", len(result))

    elif alg == "3":
        print("Using greedy algorithm to solve...")
        result = greedy_algorithm()
        endTime = time.time()

        if result:
            print_result("Greed", result)
            print("Completeness: True")
            print("Runtime: {0} second.".format(endTime - startTime))
            print("Total step: ", len(result))
        else:
            print("Completeness: False")
            print("Runtime: {0} second.".format(endTime - startTime))
            print("Total step: ", len(result))

    elif alg == "4":
        print("Using the A star algorithm to solve...")
        result = aStarAlgorithm()
        endTime = time.time()

        if result:
            print_result("AStar",result)
            print("Completeness: True")
            print("Runtime: {0} second.".format(endTime - startTime))
            print("Total step: ", len(result))
        else:
            print("Completeness: False")
            print("Runtime: {0} second.".format(endTime - startTime))
            print("Total step: ", len(result))

    else:
        print("Using all algorithms to solve...")
        result_a = aStarAlgorithm()
        time_a = time.time()-startTime
        result_dfs=DFS_algorithm()
        time_dfs = time.time()-time_a-startTime
        result_bfs=BFS_algorithm()
        time_bfs = time.time()-time_a-time_dfs-startTime
        result_greed=greedy_algorithm()
        time_greed = time.time()-time_a-time_dfs-time_bfs-startTime
        completeness=0
        if result_a:
            print_result("AStar", result_a)
            completeness+=1
        if result_greed:
            print_result("Greed", result_greed)
            completeness += 1
        if result_dfs:
            print_result("DFS", result_dfs)
            completeness += 1
        if result_bfs:
            print_result("BFS", result_bfs)
            completeness+=1
        endTime = time.time()

        print("Completeness: {0} of 4".format(completeness))
        print("TIME")
        print("A star algorithm time: {0}".format(time_a))
        print("Greedy algorithm time: {0}".format(time_greed))
        print("BFS algorithm time: {0}".format(time_bfs))
        print("DFS algorithm time: {0}".format(time_dfs))
        print("STEPS")
        print("A Star steps: ", len(result_a))
        print("Greedy steps: ", len(result_greed))
        print("BFS steps: ", len(result_bfs))
        print("DFS steps: ", len(result_dfs))
