"""
The template of the script for the machine learning process in game pingpong
"""

import numpy as np
import pickle
class MLPlay:
    def __init__(self, side):
        """
        Constructor

        @param side A string "1P" or "2P" indicates that the `MLPlay` is used by
               which side.
        """
        
        self.side = side
        """
        The main loop for the machine learning process
        The `side` parameter can be used for switch the code for either of both sides,
        so you can write the code for both sides in the same script. Such as:
        ```python
        if side == "1P":
            ml_loop_for_1P()
        else:
            ml_loop_for_2P()
        ```
        @param side The side which this script is executed for. Either "1P" or "2P".
        """
        H = 200
        D = 8
        resume = False # resume from previous checkpoint?

        if resume:
            self.model = pickle.load(open('save.p', 'rb'))
        else:
            self.model = {}
            self.model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
            self.model['W2'] = np.random.randn(H) / np.sqrt(H)

        self.grad_buffer = { k : np.zeros_like(v) for k,v in self.model.items() } # update buffers that add up gradients over a batch
        self.rmsprop_cache = { k : np.zeros_like(v) for k,v in self.model.items() } # rmsprop memory

        self.xs,self.hs,self.dlogps,self.drs = [],[],[],[]
        self.running_reward = None
        self.reward_sum = 0
        self.episode_number = 0

        self.batch_size = 10 # every how many episodes to do a param update?
        self.learning_rate = 1e-4
        self.gamma = 0.99 # discount factor for reward
        self.decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2

        # === Here is the execution order of the loop === #
        # 1. Put the initialization code here
        self.ball_served = False

    def update(self, scene_info):
        """
        Generate the command according to the received scene information
        """
        
        # if scene_info["status"] != "GAME_ALIVE":
            # return "RESET"

        # if not self.ball_served:
            # self.ball_served = True
            # return "SERVE_TO_LEFT"
        # else:
            # return "MOVE_LEFT"
            
         # 2. Inform the game process that ml process is ready
        #comm.ml_ready()
        if scene_info["status"] != "GAME_ALIVE":
            self.reset()
            return "RESET"
        count = 0
        _score = [0,1]
        _game_over_score = 11
        # 3. Start an endless loop
        while True:
            # 3.1. Receive the scene information sent from the game process
            #scene_info = comm.recv_from_game()
            # 3.2. If either of two sides wins the game, do the updating or
            #      resetting stuff and inform the game process when the ml process
            #      is ready.
            #print("1P","helloword ")

            # 3.3 Put the code here to handle the scene information
            
            # 3.4 Send the instruction for this frame to the game process
            if not self.ball_served:
                #comm.send_to_game({"frame": scene_info["frame"], "command": "SERVE_TO_LEFT"})
                #return "MOVE_LEFT"
                
                #print("1P","SERVE_TO_LEFT")
                self.ball_served = True
                return "SERVE_TO_LEFT"
            else:
                #print("1P","is SERVE_TO_LEFT")
                if self.side == "1P":
                    #print("1P policy","SERVE_TO_LEFT")
                    observation = self.getObs("1P",scene_info)
                    aprob, h = self.policy_forward(observation)
                    action = 2 if np.random.uniform() < aprob else 3

                    #print('action' + str(action))
                    #print('aprob' + str(aprob))
                    #print('a' + str(np.random.uniform()))

                    
                    # if action == 2:
                        # #comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_RIGHT"})
                        # return "MOVE_RIGHT"
                    # else:
                        # #comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_LEFT"})
                        # return "MOVE_LEFT"

                    # record various intermediates (needed later for backprop)
                    self.xs.append(observation) # observation
                    self.hs.append(h) # hidden state
                    y = 1 if action == 2 else 0 # a "fake label"
                    self.dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
                    
                    if scene_info['status'] == 'GAME_ALIVE':
                        reward = 0
                        done = False
                    elif scene_info['status'] == 'GAME_1P_WIN':
                        reward = 2
                        _score[0] += 1
                    elif scene_info['status'] == 'GAME_DRAW':
                        reward = 1
                        _score[0] += 1
                        _score[1] += 1
                    else:
                        reward = -1
                        _score[1] += 1
      
                    if _score[0] == _game_over_score or _score[1] == _game_over_score:
                        done = True
                    else:
                        done = False

                    self.reward_sum += reward
                    self.drs.append(reward)

                    if done: # an episode finished
                        self.episode_number += 1
                        _score = [0,1]
                        # stack together all inputs, hidden states, action gradients, and rewards for this episode
                        epx = np.vstack(self.xs)
                        eph = np.vstack(self.hs)
                        epdlogp = np.vstack(self.dlogps)
                        epr = np.vstack(self.drs)
                        self.xs,self.hs,self.dlogps,self.drs = [],[],[],[] # reset array memory


                        # compute the discounted reward backwards through time
                        discounted_epr = self.discount_rewards(epr)
                        discounted_epr = discounted_epr.astype('float')
                        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
                        
                        discounted_epr -= np.mean(discounted_epr).astype('float')
                        discounted_epr /= np.std(discounted_epr).astype('float')

                        epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
                        grad = self.policy_backward(eph, epdlogp)
                        for k in self.model: self.grad_buffer[k] += grad[k] # accumulate grad over batch
                        # perform rmsprop parameter update every self.batch_size episodes
                        if self.episode_number % self.batch_size == 0:
                            for k,v in self.model.items():
                                g = self.grad_buffer[k] # gradient
                                self.rmsprop_cache[k] = self.decay_rate * self.rmsprop_cache[k] + (1 - self.decay_rate) * g**2
                                model[k] += self.learning_rate * g / (np.sqrt(self.rmsprop_cache[k]) + 1e-5)
                                self.grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

                        # boring book-keeping
                        self.running_reward = self.reward_sum if self.running_reward is None else self.running_reward * 0.99 + self.reward_sum * 0.01
                        print('resetting env. episode reward total was %f. running mean: %f' % (self.reward_sum, self.running_reward))
                        if self.episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
                        self.reward_sum = 0

#                    if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
                        #print(('ep %d: game finished, reward: %f' % (self.episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))
                    command = self.ml_loop_for_1P(scene_info)
                    print("1P",command)
                else:
                    command = self.ml_loop_for_2P(scene_info)
                    print("2P",command)
                if command == 0:
                    #comm.send_to_game({"frame": scene_info["frame"], "command": "NONE"})
                    return "NONE"
                elif command == 1:
                    #comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_RIGHT"})
                    return "MOVE_RIGHT"
                else :
                    #comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_LEFT"})
                    return "MOVE_LEFT"
                


            # if scene_info["status"] != "GAME_ALIVE":
                # # Do some updating or resetting stuff
                # self.ball_served = False
            

    def reset(self):
        """
        Reset the status
        """
        H = 200
        D = 8
        resume = False # resume from previous checkpoint?

        if resume:
            self.model = pickle.load(open('save.p', 'rb'))
        else:
            self.model = {}
            self.model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
            self.model['W2'] = np.random.randn(H) / np.sqrt(H)

        self.grad_buffer = { k : np.zeros_like(v) for k,v in self.model.items() } # update buffers that add up gradients over a batch
        self.rmsprop_cache = { k : np.zeros_like(v) for k,v in self.model.items() } # rmsprop memory

        self.xs,self.hs,self.dlogps,self.drs = [],[],[],[]
        self.running_reward = None
        self.reward_sum = 0
        self.episode_number = 0

        self.batch_size = 10 # every how many episodes to do a param update?
        self.learning_rate = 1e-4
        self.gamma = 0.99 # discount factor for reward
        self.decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
        self.ball_served = False
    def sigmoid(self,x): 
        return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

    def getObs(self,player,scene_info):
        observation = []
        observation.append(scene_info['ball'][0])
        observation.append(scene_info['ball'][1])
        observation.append(scene_info['ball_speed'][0])
        observation.append(scene_info['ball_speed'][1])
        if 'blocker' in scene_info.keys():
            observation.append(scene_info['blocker'][0])
            observation.append(scene_info['blocker'][1])
        else:
            observation.append(200)
            observation.append(240)
        if player == '1P':
            observation.append(scene_info['platform_1P'][0])
            observation.append(scene_info['platform_1P'][1])
        if player == '2P':
            observation.append(scene_info['platform_2P'][0])
            observation.append(scene_info['platform_2P'][1])
        observation = np.array(observation)
        return observation

    def discount_rewards(self,r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
            #print(r[t])
            running_add = running_add * self.gamma + r[t]

            discounted_r[t] = running_add
        return discounted_r

    def policy_backward(self,eph, epdlogp):
        """ backward pass. (eph is array of intermediate hidden states) """
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, model['W2'])
        dh[eph <= 0] = 0 # backpro prelu
        dW1 = np.dot(dh.T, epx)
        return {'W1':dW1, 'W2':dW2}
  
    def policy_forward(self, x):
        h = np.dot(self.model['W1'], x)####
        h[h<0] = 0 # ReLU nonlinearity
        logp = np.dot(self.model['W2'], h)
        p = self.sigmoid(logp)
        return p, h # return probability of taking action 2, and hidden state
    

    def move_to(self,player,pred,scene_info) : #move platform to predicted position to catch ball 
        if player == '1P':
            if scene_info["platform_1P"][0]+20  > (pred-10) and scene_info["platform_1P"][0]+20 < (pred+10): return 0 # 不動
            elif scene_info["platform_1P"][0]+20 <= (pred-10) : return 1 # 右移
            else : return 2 # 左移
        else :
            if scene_info["platform_2P"][0]+20  > (pred-10) and scene_info["platform_2P"][0]+20 < (pred+10): return 0 # 不動
            elif scene_info["platform_2P"][0]+20 <= (pred-10) : return 1 # 右移
            else : return 2 # 左移
           
#---------------
    def ml_loop_for_1P(self,scene_info): 
        if scene_info['status'] == 'GAME_ALIVE':
            reward = 0
        elif scene_info['status'] == 'GAME_1P_WIN':
            reward = 2
        elif scene_info['status'] == 'GAME_DRAW':
            reward = 1
        else:
            reward = -1
        
        #print(reward)

        if scene_info["ball_speed"][1] > 0 : # 球正在向下 # ball goes down
            x = ( scene_info["platform_1P"][1]-scene_info["ball"][1] ) // scene_info["ball_speed"][1] # 幾個frame以後會需要接  # x means how many frames before catch the ball
            pred = scene_info["ball"][0]+(scene_info["ball_speed"][0]*x)  # 預測最終位置 # pred means predict ball landing site 
            bound = pred // 200 # Determine if it is beyond the boundary
            if (bound > 0): # pred > 200 # fix landing position
                if (bound%2 == 0) : 
                    pred = pred - bound*200                    
                else :
                    pred = 200 - (pred - 200*bound)
            elif (bound < 0) : # pred < 0
                if (bound%2 ==1) :
                    pred = abs(pred - (bound+1) *200)
                else :
                    pred = pred + (abs(bound)*200)
            return self.move_to(player = '1P',pred = pred,scene_info=scene_info)
        else : # 球正在向上 # ball goes up
            return self.move_to(player = '1P',pred = 100,scene_info=scene_info)



    def ml_loop_for_2P(self,scene_info):  # as same as 1P
        if scene_info["ball_speed"][1] > 0 : 
            return self.move_to(player = '2P',pred = 100,scene_info=scene_info)
        else : 
            x = ( scene_info["platform_2P"][1]+30-scene_info["ball"][1] ) // scene_info["ball_speed"][1] 
            pred = scene_info["ball"][0]+(scene_info["ball_speed"][0]*x) 
            bound = pred // 200 
            if (bound > 0):
                if (bound%2 == 0):
                    pred = pred - bound*200 
                else :
                    pred = 200 - (pred - 200*bound)
            elif (bound < 0) :
                if bound%2 ==1:
                    pred = abs(pred - (bound+1) *200)
                else :
                    pred = pred + (abs(bound)*200)
            return self.move_to(player = '2P',pred = pred,scene_info=scene_info)
#-----
   
