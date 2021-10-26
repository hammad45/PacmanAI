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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        score = 0
        
        currentPosition = currentGameState.getPacmanPosition()
        currentPositionFood = currentGameState.getFood().asList()
        nearestFoodDistance = min([manhattanDistance(currentPosition, food) for food in currentPositionFood])
        
        newPositionFood = newFood.asList()
        newNearestFoodDistance = [manhattanDistance(newPos, food) for food in newPositionFood]
        if newNearestFoodDistance:
            newNearestFoodDistance = min(newNearestFoodDistance)
        else:
            newNearestFoodDistance = 0
        foodLocation = nearestFoodDistance - newNearestFoodDistance

        scoreDiff = successorGameState.getScore() - currentGameState.getScore()

        minGhostDistance = []
        for ghost in newGhostStates:
            minGhostDistance.append(util.manhattanDistance(newPos, ghost.getPosition()))
        
        if min(minGhostDistance) <= 1:
            score -= 100
        elif foodLocation > 0:
            score += 100
        elif scoreDiff > 0:
            score += 50
        else:
            score +=10

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
    """
    Your minimax agent (question 2)
    """

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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        
        def maximizer(state, depth):           
            legalActions = state.getLegalActions(0)
            if not legalActions or self.depth == depth or state.isLose() or state.isWin():
                return self.evaluationFunction(state)
  
            return max(minimizer(state.generateSuccessor(0, action), depth, 1) for action in legalActions)

        def minimizer(state, depth, ghostIndex):
            legalActions = state.getLegalActions(ghostIndex)
            if not legalActions or self.depth == depth or state.isLose() or state.isWin():
                return self.evaluationFunction(state)
       
            if ghostIndex < state.getNumAgents() - 1:
                return min(minimizer(state.generateSuccessor(ghostIndex, action), depth, ghostIndex + 1) for action in legalActions)
            else:
                return min(maximizer(state.generateSuccessor(ghostIndex, action), depth + 1) for action in legalActions)

        legalActions = gameState.getLegalActions(0)
        futureStates = [gameState.generateSuccessor(0, move) for move in gameState.getLegalActions(0)]
        score = [minimizer(state, 0, 1) for state in futureStates]
        maxScore  = max(score)
        maxScoreIndex = score.index(maxScore)
  
        return legalActions[maxScoreIndex]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
   
        def maximizer(state, alpha, beta, depth):
            v = -float('inf')
            tmpV = v 
            legalActions = state.getLegalActions(0)
        
            if not legalActions or self.depth == depth or state.isLose() or state.isWin():
                return self.evaluationFunction(state)

            for action in legalActions:
                tmpV = minimizer(state.generateSuccessor(0, action), alpha, beta, 1, depth)
                if tmpV > v:
                    v = tmpV
                    bestAction = action
                if v > beta:
                    return v
                alpha = max(alpha, v)
            
            if depth == 0:
                return bestAction
            return v

        def minimizer(state, alpha, beta, agentIndex, depth):
            v = float('inf')
            legalActions = state.getLegalActions(agentIndex)

            if not legalActions or self.depth == depth or state.isLose() or state.isWin():
                return self.evaluationFunction(state)

            for action in legalActions:
                if agentIndex == state.getNumAgents() - 1:
                    v = min(v, maximizer(state.generateSuccessor(agentIndex, action), alpha, beta, depth + 1))
                else:
                    v = min(v, minimizer(state.generateSuccessor(agentIndex, action), alpha, beta, agentIndex + 1, depth))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        return maximizer(gameState, float("-inf"), float("inf"), 0)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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
