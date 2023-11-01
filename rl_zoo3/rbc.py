import math
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

class RbcEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=8):
        # Constants.
        self.train_length_p = 20.0
        self.train_velocity = 1.0
        self.switch_on = (0, 250, 0)
        self.switch_off = (250, 0, 0)
        self.black = (0, 0, 0)
        self.track_color = self.black
        self.train_1_color = (0, 0, 255)
        self.train_2_color = (255, 0, 0)
        # MA colors are between train color and track color.
        self.train_1_ma_color = (0, 0, 130)
        self.train_2_ma_color = (130, 0, 0)
        self.block_margin = 5 # Number of extra outside-screen blocks to account for while computing distance ahead.
        # Max braking and acceleration force.
        self.max_B = 1.0
        self.max_A = 1.0

        self._POS = self._TRK_NO = 0
        self._VEL = self._BLK_NO = 1
        self._TRK_1 = 1
        self._TRK_2 = 2

        # Naming convention: _b indicates in terms of blocks, _p indicates in terms of pixels.
        self.block_size = size  # The size of the train blocks
        self.trackspace_p = 50    # Vertical space between tracks
        self.window_size_p = 512  # The size of the PyGame window. 1 pixel = 1 unit of distance.

        # Initialize track geometry.
        self.track1_length_b = int(self.window_size_p/self.block_size)
        self.track2_length_b = int(self.track1_length_b/2)
        self.track1_start = 0
        self.switch_1_pos_p = int(self.window_size_p/4)
        # Index of starting block on track 2.
        self.track2_start_b = self.switch_1_pos_p//self.block_size
        # Index of ending block on track 2.
        self.track2_end_b = self.track2_start_b + self.track2_length_b
        self.switch_2_pos_p = (self.track2_end_b+1)*self.block_size

        # Observations are dictionaries with train state information.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                # Train 1's position and velocity are both floats.
                "train1": spaces.Box(low=0, high=self.window_size_p, shape=(2,), dtype=np.float64),
                # Train 1's occupied blocks is an array with 1 at the block indices that train 1 occupies.
                "train1_occupied": spaces.MultiDiscrete([2] * (self.track1_length_b*2), dtype=np.int32),
                # Train 2's position and velocity are both floats.
                "train2": spaces.Box(low=0, high=self.window_size_p, shape=(2,), dtype=np.float64),
                # Train 2's occupied blocks is an array with 1 at the block indices that train 2 occupies.
                "train2_occupied": spaces.MultiDiscrete([2] * (self.track1_length_b*2), dtype=np.int32),
                # Train 1's motion authority is a list, where
                # 1 means the block at that index is in ma1 and 0 means it is not.
                "ma1": spaces.MultiDiscrete([2] * (self.track1_length_b*2), dtype=np.int32),
                # Train 2's motion authority is similar.
                "ma2": spaces.MultiDiscrete([2] * (self.track1_length_b*2), dtype=np.int32),
                # Switch position is a list of 0 or 1, where 0 is off, so no track change and 1 is on, so track changes.
                "switches": spaces.MultiDiscrete([2] * 2),
            }
        )

        # The action space consists of 4 numbers.
        # 1. The start of the motion authority for train 1.
        # 2. The end of the motion authority for train 1 (ma includes this block).
        # 3. The start of the motion authority for train 2.
        # 4. The end of the motion authority for train 2 (ma includes this block).
        self.action_space = spaces.MultiDiscrete([self.track1_length_b]*4)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def get_block_before(self, pixel):
        return int(pixel//self.block_size)
        
    def get_block_after(self, pixel):
        return int(math.ceil(pixel/self.block_size))

    def get_pixel_at_end(self, block):
        return int((block+1)*self.block_size)
    
    def get_ma1(self, start, end):
        if start>end:
            return []
        ma = []
        for i in range(start, end+1):
            track = self._TRK_1
            if (i>=self.track2_start_b and i<=self.track2_end_b):
                track = self._TRK_2
            ma.append([track, i])
        return np.array(ma)
    
    def get_ma2(self, start, end):
        if start>end:
            return []
        ma = []
        for i in range(start, end+1):
            ma.append([self._TRK_1, i])
        return np.array(ma)
    
    def blocks_to_sparse_vec(self, blocks):
        """Converts a list of blocks to a spaces.MultiDiscrete([2] * (self.track1_length_b*2))
        with 1 indicating the block is in the list, and 0, that it is not."""
        result = (np.zeros(dtype=np.int32, shape=(self.track1_length_b*2)))
        for block in blocks:
            if (block[self._BLK_NO]<0 or block[self._BLK_NO]>=self.track1_length_b):
                continue
            index = (block[self._TRK_NO]-1)*self.track1_length_b + block[self._BLK_NO]
            result[index] = 1
        return result

    def _get_obs(self):
        obs_dict = {
            "train1": self.train_1,
            "train1_occupied": self.blocks_to_sparse_vec(self.train_1_occupied),
            "train2": self.train_2,
            "train2_occupied": self.blocks_to_sparse_vec(self.train_2_occupied),
            "ma1": self.blocks_to_sparse_vec(self.ma1),
            "ma2": self.blocks_to_sparse_vec(self.ma2),
            "switches": self.switches
        }
        return obs_dict
        # return gym.wrappers.FlattenObservation(obs_dict)

    def _get_info(self):
        return self._get_obs()

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Train 1 starts on track 1 block 2.
        self.train_1 = np.array([16.0, self.train_velocity])
        # Train 1 occupies the first block.
        self.train_1_occupied = np.array([[1,-2], [1,-1], [1, 0]])

        # Train 2 starts at track 1 end, with a small offset.
        self.train_2 = np.array([float(self.get_pixel_at_end(self.track1_length_b)-16.0), -self.train_velocity])
        # Train 2 occupies the last block.
        self.train_2_occupied = np.array([[1, self.track1_length_b-1], [1, self.track1_length_b]])

        # # Train 1's motion authority is the first block.
        # self.ma1 = np.array([[1, 0]])
        self.ma1 = [[1, -4], [1,-3], [1, -2], [1, -1], [1, 0], [1,1], [1,2], [1,3], [1,4], [1,5], [1,6], [1,7]]
        # # Train 2's motion authority is the last block.
        # self.ma2 = np.array([[1, self.track1_length-5]])
        self.ma2 = [[1, self.track1_length_b-8], [1, self.track1_length_b-7], [1, self.track1_length_b-6], [1, self.track1_length_b-5], [1, self.track1_length_b-4], [1, self.track1_length_b-3], [1, self.track1_length_b-2], [1, self.track1_length_b-1], [1, self.track1_length_b], [1, self.track1_length_b+1]]

        # Switches are both on.
        self.switches = np.array([1, 1])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def get_fwd_space(self, current, ma, startpt):
        """For train 1, distance ot the end of motion authority."""
        ma = [list(b) for b in ma]
        if (list(current) not in ma):
            return 0
        offset = startpt%self.block_size
        dist = self.block_size-offset
        if (offset==0):
            dist = 0
        block_after = self.get_block_after(startpt)
        for i in range (block_after, self.get_block_after(self.switch_1_pos_p)):
            if [self._TRK_1, i] in ma:
                dist += self.block_size
            else:
                return dist
        for i in range (max(block_after, self.get_block_after(self.switch_1_pos_p)), self.get_block_after(self.switch_2_pos_p)):
            if [self._TRK_2, i] in ma:
                dist += self.block_size
            else:
                return dist
        for i in range (max(block_after,self.get_block_after(self.switch_2_pos_p)), self.track1_length_b):
            if [self._TRK_1, i] in ma:
                dist += self.block_size
            else:
                return dist
        return dist

    def get_bwd_space(self, ma, startpt):
        """For train 2, distance going backwards over the blocks to the end of motion authority."""
        ma = [list(b) for b in ma]
        offset = startpt%self.block_size
        dist = offset
        block_before = self.get_block_before(startpt)
        if ([self._TRK_1, block_before] not in ma):
            return 0    
        for i in range(block_before-1, -self.block_margin, -1):
            if [self._TRK_1, i] in ma:
                dist += self.block_size
            else:
                return dist
        return dist
        
    # Train physics.

    def braking_distance(self, velocity):
        """If we start braking now, distance that train travels before it halts."""
        return (velocity**2)/(2*self.max_B)
    
    def acc_cycle_distance(self, velocity):
        """Distance covered in a single time period of acceleration."""
        assert velocity>=0
        return velocity + 0.5*self.max_A
    
    def brake_cycle_distance(self, velocity):
        """Distance covered in a single time period of braking."""
        assert velocity>=0
        if (velocity/self.max_B>=1):
            return velocity - 0.5*self.max_B
        else:
            return self.braking_distance(velocity)

    def stopping_distance(self, velocity):
        """Returns stopping distance: distance train travels through one time 
        unit of acceleration and then braking to a stop."""
        assert velocity>=0
        return self.acc_cycle_distance(velocity) + self.braking_distance(velocity+self.max_A)

    def step(self, action):
        self.ma1 = self.get_ma1(action[0], action[1])
        self.ma2 = self.get_ma2(action[2], action[3])

        # Time elapsed is one unit.
        # Right now, no acceleration. Train advances with train_velocity if it has the space to do so.
        # If it doesn't have the space, it stops.
        train_pos = self.train_1[self._POS] # Absolute distance that train traveled.
        # Does not identify track, but mod blocksize, gives offset from start of block, train_pos=0 is aligned with the start of some block.
        last_block = self.train_1_occupied[-1]
        dist_ahead = self.get_fwd_space(last_block, self.ma1, train_pos)
        
        # Case 1: train has space to advance for the full time period.
        if (self.stopping_distance(self.train_1[self._VEL])<dist_ahead):
            self.train_1[self._POS] = self.acc_cycle_distance(self.train_1[self._VEL])+train_pos
            self.train_1[self._VEL] = self.train_1[self._VEL] + self.max_A
        # Case 2: train doesn't have space to advance for the full time period. Just advance to the end of authority.
        else:
            self.train_1[self._POS] = self.brake_cycle_distance(self.train_1[self._VEL])+train_pos
            self.train_1[self._VEL] = max(self.train_1[self._VEL] - self.max_B, 0)
        # print("Train 1 position (after): ", self.train_1[self._POS])

        new_train_occupied = []
        first_block = self.get_block_before(self.train_1[self._POS])
        last_block = self.get_block_before(self.train_1[self._POS]-self.train_length_p)
        for i in range(last_block, first_block+1):
            new_track = self._TRK_1
            if (i>=self.track2_start_b and i<=self.track2_end_b):
                new_track = self._TRK_2
            new_train_occupied.append([new_track, i])
        self.train_1_occupied = np.array(new_train_occupied)

        # Train 2
        train_pos = self.train_2[self._POS]
        dist_ahead = self.get_bwd_space(self.ma2, train_pos)
        # Case 1: train has space to advance for the full time period.
        if (self.stopping_distance(-self.train_2[self._VEL])<dist_ahead):
            self.train_2[self._POS] = train_pos-self.acc_cycle_distance(-self.train_2[self._VEL])
            self.train_2[self._VEL] = self.train_2[self._VEL] - self.max_A
        # Case 2: train doesn't have space to advance for the full time period. Just advance to the end of authority.
        else:
            self.train_2[self._POS] = train_pos-self.brake_cycle_distance(-self.train_2[self._VEL])
            self.train_2[self._VEL] = min(self.train_2[self._VEL] + self.max_B, 0)
        # print("Train 2 position (after): ", self.train_2[self._POS])

        new_train_occupied = []
        first_block = self.get_block_before(self.train_2[self._POS])
        last_block = self.get_block_before(self.train_2[self._POS]+self.train_length_p)
        for i in range(first_block, last_block+1):
            new_train_occupied.append([self._TRK_1, i])
        self.train_2_occupied = np.array(new_train_occupied)

        # An episode is done iff both the trains have exited and occupy no blocks.
        filtered_1 = list(filter(lambda block : block[self._BLK_NO]>=0 and block[self._BLK_NO]<=self.track1_length_b, self.train_1_occupied))
        filtered_2 = list(filter(lambda block : block[self._BLK_NO]>=0 and block[self._BLK_NO]<=self.track2_length_b, self.train_2_occupied))
        terminated = len(filtered_1) == 0 and len(filtered_2) == 0
        # Penalty for large motion authorities.
        ma_size_penalty = - len(self.ma1) - len(self.ma2)
        # Penalty for trains outside MAs.
        ma1 = [list(b) for b in self.ma1]
        ma2 = [list(b) for b in self.ma2]
        train_1_occ = [list(b) for b in self.train_1_occupied]
        train_2_occ = [list(b) for b in self.train_2_occupied]
        train1_outside_ma = list(filter(lambda block : block not in ma1, train_1_occ))
        train2_outside_ma = list(filter(lambda block : block not in ma2, train_2_occ))
        ma_train_penalty = (-len(train1_outside_ma) - len(train2_outside_ma))*10
        # Penalty for intersecting MAs.
        ma1_set = set(map(tuple, ma1))
        ma2_set = set(map(tuple, ma2))
        intersect = ma1_set.intersection(ma2_set)
        intersect_penalty = -len(intersect)*100
        # Reward for velocity.
        velocity_reward = (abs(self.train_1[self._VEL]) + abs(self.train_2[self._VEL]))*100
        # Scaling to try to keep the total between -5 and 5.
        scaling_factor = 1000000.
        # Reward for terminating \ penalty otherwise.
        reward = 5 if terminated \
            else (ma_size_penalty+ma_train_penalty+intersect_penalty+velocity_reward)/scaling_factor
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    # Drawing related functions.

    def get_block_rect(self, track_num, block_num):
        return pygame.Rect(
            self.block_size * block_num + self.track1_start,
            self.window_size_p/2 - self.trackspace_p if track_num == 1 else self.window_size_p/2 + self.trackspace_p,
            self.block_size-1,
            self.trackspace_p/2,
        )

    def draw_arrow(
            self,
            surface: pygame.Surface,
            start: pygame.Vector2,
            end: pygame.Vector2,
            color: pygame.Color,
            body_width: int = 2,
            head_width: int = 10,
            head_height: int = 14,
        ):
        """Draw an arrow between start and end with the arrow head at the end.
        From https://www.reddit.com/r/pygame/comments/v3ofs9/draw_arrow_function/.

        Args:
            surface (pygame.Surface): The surface to draw on
            start (pygame.Vector2): Start position
            end (pygame.Vector2): End position
            color (pygame.Color): Color of the arrow
            body_width (int, optional): Defaults to 2.
            head_width (int, optional): Defaults to 4.
            head_height (float, optional): Defaults to 2.
        """
        arrow = start - end
        angle = arrow.angle_to(pygame.Vector2(0, -1))
        body_length = arrow.length() - head_height

        # Create the triangle head around the origin
        head_verts = [
            pygame.Vector2(0, head_height / 2),  # Center
            pygame.Vector2(head_width / 2, -head_height / 2),  # Bottomright
            pygame.Vector2(-head_width / 2, -head_height / 2),  # Bottomleft
        ]
        # Rotate and translate the head into place
        translation = pygame.Vector2(0, arrow.length() - (head_height / 2)).rotate(-angle)
        for i in range(len(head_verts)):
            head_verts[i].rotate_ip(-angle)
            head_verts[i] += translation
            head_verts[i] += start

        pygame.draw.polygon(surface, color, head_verts)

        # Stop weird shapes when the arrow is shorter than arrow head
        if arrow.length() >= head_height:
            # Calculate the body rect, rotate and translate into place
            body_verts = [
                pygame.Vector2(-body_width / 2, body_length / 2),  # Topleft
                pygame.Vector2(body_width / 2, body_length / 2),  # Topright
                pygame.Vector2(body_width / 2, -body_length / 2),  # Bottomright
                pygame.Vector2(-body_width / 2, -body_length / 2),  # Bottomleft
            ]
            translation = pygame.Vector2(0, body_length / 2).rotate(-angle)
            for i in range(len(body_verts)):
                body_verts[i].rotate_ip(-angle)
                body_verts[i] += translation
                body_verts[i] += start

            pygame.draw.polygon(surface, color, body_verts)

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size_p, self.window_size_p))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size_p, self.window_size_p))
        canvas.fill((255, 255, 255))
       
        # First we draw the tracks.
        # We draw the tracks as a sequence of rectangles.

        # Track 1:
        for i in range(self.track1_length_b):
            pygame.draw.rect(
                canvas,
                self.black,
                self.get_block_rect(1, i)
            )
        
        # Track 2:
        for i in range(self.track2_start_b, self.track2_end_b+1):
            pygame.draw.rect(
                canvas,
                self.black,
                self.get_block_rect(2, i)
            )
        
        # Overwrite the blocks that belong to a train's MA with the corresponding color.
        # MA1.
        for block in self.ma1:
            track_num, block_num = block
            pygame.draw.rect(
                canvas,
                self.train_1_ma_color,
                self.get_block_rect(track_num, block_num)
            )
        
        # MA2.
        for block in self.ma2:
            track_num, block_num = block
            pygame.draw.rect(
                canvas,
                self.train_2_ma_color,
                self.get_block_rect(track_num, block_num)
            )

        # Switch 1:
        self.draw_arrow(
            canvas,
            pygame.Vector2(self.switch_1_pos_p, self.window_size_p/2 - self.trackspace_p/2),
            pygame.Vector2(self.switch_1_pos_p, self.window_size_p/2 + self.trackspace_p),
            self.switch_on if self.switches[0] else self.switch_off,
        )

        # Switch 2:
        self.draw_arrow(
            canvas,
            pygame.Vector2(self.switch_2_pos_p, self.window_size_p/2 + self.trackspace_p),
            pygame.Vector2(self.switch_2_pos_p, self.window_size_p/2 - self.trackspace_p/2),
            self.switch_on if self.switches[1] else self.switch_off,
        )

        # Next we draw the trains.
        # We draw the trains as a sequence of rectangles.
        for block in self.train_1_occupied:
            track_num, block_num = block
            pygame.draw.rect(
                canvas,
                self.train_1_color,
                self.get_block_rect(track_num, block_num)
            )

        for block in self.train_2_occupied:
            track_num, block_num = block
            pygame.draw.rect(
                canvas,
                self.train_2_color,
                self.get_block_rect(track_num, block_num)
            )


        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()