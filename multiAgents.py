# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.

        Idea here is to reduce the score of a state if pacman is far from the food.
        Also if pacman is very near to ghost that also will have a negative impact on score
        Num of food points left will also affect the score. More the food points left, lesser
         the score will be.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        num_food = successorGameState.getNumFood()+1;
        newGhostStates = successorGameState.getGhostStates()
        ghostPosition = successorGameState.getGhostPositions()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        min_dist = 999999

        #to find the corner cordinates of the grid
        top = successorGameState.getWalls().height
        right = successorGameState.getWalls().width

        # This loop is to find the minimum distance of ghost from pacman.
        # For loop is used for generalizing, in case there are more than one ghost
        for ghost in ghostPosition:
            dist = abs(ghost[0]-newPos[0]) + abs(ghost[1]-newPos[1])
            if dist < min_dist:
                min_dist = dist

        # This loop is to determine the nearest food distance from the Pacman position.
        food_min_dist = 999999.0
        for i in range(0,(right-1)):
            for j in range(0,(top-1)):
                if newFood[i][j] is True:
                    dist = abs(newPos[0]-i) + abs(newPos[1]-j)
                    if dist<food_min_dist:
                        food_min_dist = dist
        if food_min_dist == 999999.0:
            food_min_dist = 1.0

        # if the ghost are more than 6steps away then not considering any effect of the ghost
        # if distance between pacman and ghost is 0, then a big negative value is assigned
        if min_dist<6:
            ghostComponent = -9999 if min_dist==0 else -(10.0/min_dist)
        else:
            ghostComponent = 0

        # score will be the evaluation of the state.
        score = ghostComponent + (10.0/food_min_dist) + 1.0/num_food +  successorGameState.getScore()
        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    nodecount=0
    """
      Your minimax agent (question 2)
    """

    """--------------MINMAX implementation starts------------------------------------"""

    """we have defined two functions for implementing minmax algorithm
    1.pacman(gameState,index,depth) : it is the function for the MAX player or the pacman to determine the best possible move
                                        or score at a given possible state. Returns the best score or the optimal move from a state
    2.ghostFunc(gameState,index,depth): it is the function for the MIN player or the ghosts to determine the least possible score
                                        for the PACMAN. It returns the least possible score at a given game state"""

    def pacman(self, gameState, index, depth):
        """Check if the state is a terminal state or the state has reached the provided depth.
        If yes then return the state value"""
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)

        """if not terminal state or not reached the depth then continue with below steps.
                 1. find the legal actions"""

        actions = gameState.getLegalActions(index)

        """      2. iterate through the next states(MIN state and MAX state) recursively in depth first manner 
        to find the best possible score for max state"""

        max_score = float("-inf")
        score = max_score
        best_action = Directions.STOP
        for action in actions:
            if action != Directions.STOP :

                """3.Generate the next state(MINSTATE)"""
                nextminstate = gameState.generateSuccessor(index, action)
                self.nodecount=self.nodecount + 1

                """4.Pass the MINSTATE to the ghostFunc to obtain a score.
                     The ghostFunc always returns the least possbile score"""
                score = self.ghostFunc(nextminstate, index + 1, depth)
                if score > max_score:
                    max_score = score
                    best_action = action
        """5. If the recursion is at depth 0 then return the optimal move else return the best score obtained from all MINSTATES"""
        if depth == 0:
            return best_action
        else:
            return max_score

    def ghostFunc(self, gameState, index, depth):
        """Check if the state is a terminal state.
                If yes then return the state value"""
        if gameState.isLose() or gameState.isWin() :
            return self.evaluationFunction(gameState)

        """if not terminal state or not reached the depth then continue with below steps.
                1. find the legal actions"""
        actions = gameState.getLegalActions(index)

        min_score = float("inf")
        score=min_score

        next_index = index + 1

        """      2. iterate through the next states(MIN state and MAX state) recursively in depth first manner 
                to find the best possible score for max state"""
        for action in actions:
            if action != Directions.STOP :
                """3.Generate the next state"""
                nextstate = gameState.generateSuccessor(index, action)
                self.nodecount = self.nodecount + 1
                """If index is 1 less than total number of agents, then all ghosts have been moved."""
                """4. Pass the nextstate to pacman function to get a score."""
                if index == gameState.getNumAgents() - 1:
                    score = self.pacman(nextstate, 0, depth + 1)
                else:
                    """5. If index < total number of agents - 1 then pass the next state to ghostFunc with index + 1(next ghost)"""
                    score = self.ghostFunc(nextstate, next_index, depth)
            if score < min_score:
                min_score = score
        return min_score

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        """the pacman funtion is called in the beginning to get the best possible move for the pacman"""

        path= self.pacman(gameState,self.index ,0)
        print self.nodecount
        return path
    """-----------------MINMAX implementation ends------------------------------------"""

class AlphaBetaAgent(MultiAgentSearchAgent):
    nodecount = 0
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def pacman(self, gameState, index, depth, alpha, beta):

        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(index)

        max_score = float("-inf")
        score = max_score
        best_action = Directions.EAST
        for action in actions:
            if action != Directions.STOP:
                nextminstate = gameState.generateSuccessor(index, action)
                self.nodecount=self.nodecount + 1
                score = self.ghostFunc(nextminstate, index + 1, depth, alpha, beta)
                if score > max_score:
                    max_score = score
                    best_action = action
                alpha = max(max_score, alpha)
            if max_score > beta:
                break
        if depth == 0:
            return best_action
        else:
            return max_score

    def ghostFunc(self, gameState, index, depth, alpha, beta):
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(index)
        min_score = float("inf")

        score = min_score

        next_index = index + 1

        for action in actions:
            if action != Directions.STOP:
                nextstate = gameState.generateSuccessor(index, action)
                self.nodecount = self.nodecount + 1
                if index == gameState.getNumAgents() - 1:
                    score = self.pacman(nextstate, 0, depth + 1, alpha, beta)
                else:
                    score = self.ghostFunc(nextstate, next_index, depth, alpha, beta)
            if score < min_score:
                min_score = score
            beta = min(beta, min_score)
            if min_score < alpha:
                break

        return min_score

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        #First max(pacman) move is performed in the below for loop
        # Notion of depth here is kind of considered as height. The root is at self.depth and then
        # depth reduces with subsequent ply of game
        path = self.pacman(gameState, self.index, 0, float("-inf"), float("inf"))
        print self.nodecount
        return path

class ExpectimaxAgent(MultiAgentSearchAgent):
    nodecount = 0
    """
      Your expectimax agent (question 4)
    """

    def is_terminal(self, depth, state):
        """This function tells you whether we reached terminal state or not"""
        if depth == 0 or state.isLose() or state.isWin():
            return True
        return False

    def decide_func(self, agent_number, state):
        """
        This function decide whether to call max_value or min_value based on the agent
        index whose turn will come next
         """
        if agent_number == 0:
            return self.max_value
        else:
            return self.chance_value

    def max_value(self, state, depth, agent_number):
        depth -= 1
        if self.is_terminal(depth, state):
            return self.evaluationFunction(state)
        alpha_value = float("-inf")

        for action in state.getLegalActions((agent_number % state.getNumAgents())):
            # call_func will be the function we need to call according to agent index
            # decide_func tells which function to call based on agent index
            call_func = self.decide_func(((agent_number + 1) % state.getNumAgents()), state)
            # below_min gives the min from the remaining lower part of the three
            self.nodecount = self.nodecount + 1
            below_min = call_func(state.generateSuccessor((agent_number % state.getNumAgents()), action), depth,
                                  ((agent_number + 1) % state.getNumAgents()))
            alpha_value = max(alpha_value, below_min)

        return alpha_value

    def chance_value(self, state, depth, agent_number):
        if self.is_terminal(depth, state):
            return self.evaluationFunction(state)
        beta_value = 0.0
        legal_actions = state.getLegalActions((agent_number % state.getNumAgents()))
        for action in legal_actions:
            # call_func will be the function we need to call according to agent index
            # decide_func tells which function to call based on agent index
            call_func = self.decide_func(((agent_number + 1) % state.getNumAgents()), state)
            self.nodecount = self.nodecount + 1
            below_max = call_func(state.generateSuccessor((agent_number % state.getNumAgents()), action), depth,
                                  ((agent_number + 1) % state.getNumAgents()))
            beta_value += below_max

        # since all actions are equally probable, returning the average score
        return beta_value/len(legal_actions)

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        #First max(pacman) move is performed in the below for loop
        # Notion of depth here is kind of considered as height. The root is at self.depth and then
        # depth reduces with subsequent ply of game
        max_score = float("-inf")
        max_score_action = Directions.STOP
        for action in gameState.getLegalActions(self.index):
            depth = self.depth
            # call_func will be the function we need to call according to agent index
            # decide_func tells which function to call based on agent index
            call_func = self.decide_func(self.index + 1, gameState)
            self.nodecount = self.nodecount + 1
            score = call_func(gameState.generateSuccessor(self.index, action), depth, self.index + 1)
            if score > max_score:
                max_score = score
                max_score_action = action
        print self.nodecount
        return max_score_action



def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

