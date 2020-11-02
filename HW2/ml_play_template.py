"""The template of the main script of the machine learning process
"""

import games.arkanoid.communication as comm
from games.arkanoid.communication import ( \
    SceneInfo, GameInstruction, GameStatus, PlatformAction
)

import numpy as np
import pickle
def ml_loop():
    """The main loop of the machine learning process
    This loop is run in a separate process, and communicates with the game process.
    Note that the game process won't wait for the ml process to generate the
    GameInstruction. It is possible that the frame of the GameInstruction
    is behind of the current frame in the game process. Try to decrease the fps
    to avoid this situation.
    """

    # === Here is the execution order of the loop === #
    # 1. Put the initialization code here.
    ball_position_history=[]
    filename='knn_example.sav'
    model=pickle.load(open(filename, 'rb'))
    # 2. Inform the game process that ml process is ready before start the loop.
    comm.ml_ready()

    # 3. Start an endless loop.
    while True:
        # 3.1. Receive the scene information sent from the game process.
        scene_info = comm.get_scene_info()
        ball_position_history.append(scene_info.ball)
        if len(ball_position_history) > 2:
            inp_temp=np.array([ball_position_history[-2][0], ball_position_history[-2][1], ball_position_history[-1][0], ball_position_history[-1][1], scene_info.platform[0]])
            input=inp_temp[np.newaxis, :]
            print(input)
        
            if scene_info.status == GameStatus.GAME_OVER or \
                scene_info.status == GameStatus.GAME_PASS:
                comm.ml_ready()
                continue
        
            move=model.predict(input)
            print(move)
            if move > 0:
                comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)
            elif move < 0:
                comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
            else:
                comm.send_instruction(scene_info.frame, PlatformAction.NONE)
        # 3.2. If the game is over or passed, the game process will reset
        #      the scene and wait for ml process doing resetting job.
        
            # Do some stuff if needed

            # 3.2.1. Inform the game process that ml process is ready
            #scene_info = comm.get_scene_info()

        # 3.3. Put the code here to handle the scene information
