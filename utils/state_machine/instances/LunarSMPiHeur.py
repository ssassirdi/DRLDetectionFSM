from utils.state_machine.common.Politic import Politic
from utils.state_machine.common.StateMachine import StateMachine

class StateMachineLunarLander(StateMachine):
    #State 0 : High
    #State 1 : Instable
    #State 2 : Low
    #State 3 : Grounded
    
    def transition(self, action, observation):
        x, y, xDot, yDot, theta, thetaDot, left, right = observation[0], observation[1], observation[2], observation[3], observation[4], observation[5], observation[6], observation[7]
        left = abs(left) > 0.01
        right = abs(right) > 0.01
        if self.currentState == 0:
            if abs(thetaDot) > 0.2:
                self.currentState = 1
            else:
                if y < 1:
                    self.currentState = 2
                
        elif self.currentState == 1 :
            if abs(thetaDot) < 0.2:
                if y < 1:
                    self.currentState = 2
                else:
                    self.currentState = 0
                
        elif self.currentState == 2:
            if abs(theta) > 0.1:
                self.currentState = 1
            elif left or right:
                self.currentState = 3
            elif y > 1:
                self.currentState = 0
        
        elif self.currentState == 3:
            if not(left or right):
                self.currentState = 2
                
class High(Politic):
    def __init__(self):
        pass
    
    def act(self, observation):
        return 0
    
class Instable(Politic):
    def __init__(self):
        pass
    
    def act(self, observation):
        if observation[5] > 0:  
            return 3
        else:
            return 1

class Low(Politic):
    def __init__(self):
        pass
    
    def act(self, observation):
        return 2

class Grounded(Politic):
    def __init__(self):
        pass
    
    def act(self, observation):
        return 0