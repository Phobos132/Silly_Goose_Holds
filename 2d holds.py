import cadquery as cq
import random as rnd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import copy
#import sympy as sym

# This script generates a 2d profile climbing hold that can be cut from
# a 2x4

def cross2d(x, y):
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

class arc:
    #points = pd.DataFrame(index=['start','end','center'],columns=['x','y'],dtype=float)
    #clockwise = False
    #radius = 0.0
    def __init__(self,start_point = [0,0]
                 ,end_point = [1,1],
                 center_point = [1,0],
                 clockwise_in = True):
        self.points = pd.DataFrame(index=['start','end','center'],columns=['x','y'],dtype=float)
        self.points.loc['start'] = start_point
        self.points.loc['end'] = end_point
        self.points.loc['center'] = center_point
        self.clockwise = clockwise_in
        self.refresh()
    
    def reverse(self):
        reversed_arc = arc(start_point = self.points.loc['end'],
                           end_point = self.points.loc['start'],
                           center_point = self.points.loc['center'],
                           clockwise_in = not self.clockwise)
        return reversed_arc
    
    def refresh(self):
        self.radius = self.check_radius()
        self.points.loc['midpoint'] = self.find_midpoint()
        
    def check_radius(self):
        if (np.linalg.norm(self.points.loc['start'] 
                           - self.points.loc['center'])
            - np.linalg.norm(self.points.loc['end'] 
                              - self.points.loc['center'])) > 1e-10:
            raise Exception("This is from the 'arc' class: the distance from "
                            "the center to the end and the center to the start"
                            " are different!")
        else:
            radius = np.linalg.norm(self.points.loc['start'] 
                                          - self.points.loc['center'])
        return radius
    
    def get_tangent_angle(self, point = 'start'):
        vector = self.points.loc[point] - self.points.loc['center']
        vector_angle = np.arctan2(vector['y'],vector['x'])
        if self.clockwise:
            tangent_angle = vector_angle - np.pi/2
        else:
            tangent_angle = vector_angle + np.pi/2
        return tangent_angle
    
    def plot_arc(self,axes):
        start_vector = self.points.loc['start'] - self.points.loc['center']
        start_angle = np.arctan2(start_vector['y'],start_vector['x'])*180.0/np.pi % 360.0
        end_vector = self.points.loc['end'] - self.points.loc['center']
        end_angle = np.arctan2(end_vector['y'],end_vector['x'])*180.0/np.pi % 360.0
        if self.clockwise:
            arc = patch.Arc(self.points.loc['center'],self.radius*2,self.radius*2, theta2 = start_angle, theta1 = end_angle)
        else:
            arc = patch.Arc(self.points.loc['center'],self.radius*2,self.radius*2, theta1 = start_angle, theta2 = end_angle)
            #axes.add_artist(edge_arc)
        axes.add_patch(arc)
    
    def find_midpoint(self):
        start_vector = self.points.loc['start'] - self.points.loc['center']
        start_angle = np.arctan2(start_vector['y'],start_vector['x'])
        end_vector = self.points.loc['end'] - self.points.loc['center']
        end_angle = np.arctan2(end_vector['y'],end_vector['x'])
        #make it so the end angle is always greater than the start to make it
        #easier to deal with clockwise/counter clockwise stuff
        if (end_angle < start_angle):
            end_angle = end_angle + np.pi*2
        mid_angle = ((start_angle + end_angle) / 2)
        if self.clockwise:
            mid_angle = mid_angle + np.pi
        
        midpoint = self.points.loc['center'] + [self.radius*np.cos(mid_angle),self.radius*np.sin(mid_angle)]
        return midpoint
    
    def get_arc_length(self):
        start_vector = self.points.loc['start'] - self.points.loc['center']
        start_angle = np.arctan2(start_vector['y'],start_vector['x']) % (2*np.pi)
        end_vector = self.points.loc['end'] - self.points.loc['center']
        end_angle = np.arctan2(end_vector['y'],end_vector['x']) %  (2*np.pi)
        
        if self.clockwise:
            angle = (start_angle - end_angle) % (2*np.pi)
        else:
            angle = (end_angle - start_angle)  % (2*np.pi)
        
        arc_length = angle * self.radius
        return arc_length
    
    def scale(self,scale_factor):
        scaled_arc = arc(start_point = self.points.loc['start'] * scale_factor,
                           end_point = self.points.loc['end'] * scale_factor,
                           center_point = self.points.loc['center'] * scale_factor,
                           clockwise_in = self.clockwise)
        return scaled_arc
    
my_arc = arc([0,0],[2,2],[2,0],True)
my_arc.points.loc['start','x'] = 0
my_arc.clockwise
my_arc.radius
test = my_arc.reverse()

class hold_profile:
    # class to hold all the information needed to define a hold consisting of
    #six arcs in a dictionary
    # should also have a serial number made from the values in the arcs
    
    def __init__(self,
                 top_edge_position = [30,30],
                 top_edge_radius = 10,
                 top_ledge_angle = np.pi/10,
                 top_ledge_start_height = 25,
                 bottom_edge_position = [30,-30],
                 bottom_edge_radius = 10,
                 bottom_ledge_angle = -np.pi/10,
                 bottom_ledge_start_height = -25,
                 face_angle = np.pi/2,
                 face_thickness = 30
                 ):
        
        self.arcs = pd.Series(index = [
                              'top_ledge',
                                'top_edge',
                                'top_face',
                                'bottom_face',
                                'bottom_edge',
                                'bottom_ledge'
                                ])
        
        self.arcs['top_edge'] = arc(clockwise_in = True)
        self.arcs['top_edge'].points.loc['center'] = top_edge_position
        self.arcs['top_edge'].radius = top_edge_radius
        
        start_point = pd.Series(data = [0,top_ledge_start_height],index=['x','y'],dtype=float)
        self.arcs['top_ledge'] = self.find_tangent_arc(
            start_point,
            top_ledge_angle,
            self.arcs['top_edge'].points.loc['center'],
            self.arcs['top_edge'].radius,
            "left"
            )
        
        start_point = pd.Series(data = [face_thickness,0],index=['x','y'],dtype=float)
        temp_arc = self.find_tangent_arc(
            start_point,
            face_angle,
            self.arcs['top_edge'].points.loc['center'],
            self.arcs['top_edge'].radius,
            "right"
            )
        # need to flip this arc around
        self.arcs['top_face'] = copy.deepcopy(temp_arc)
        self.arcs['top_face'].points.loc['start'] = temp_arc.points.loc['end']
        self.arcs['top_face'].points.loc['end'] = temp_arc.points.loc['start']
        self.arcs['top_face'].clockwise = not temp_arc.clockwise
        self.arcs['top_face'].refresh()
        
        self.arcs['top_edge'].points.loc['start'] = self.arcs['top_ledge'].points.loc['end']
        self.arcs['top_edge'].points.loc['end'] = self.arcs['top_face'].points.loc['start']
        self.arcs['top_edge'].refresh()

        self.arcs['bottom_edge'] = arc(clockwise_in = True)
        self.arcs['bottom_edge'].points.loc['center'] = bottom_edge_position
        self.arcs['bottom_edge'].radius = bottom_edge_radius
        
        start_point = pd.DataFrame(index=['start'],columns=['x','y'],dtype=float)
        start_point.loc['start'] = [0,bottom_ledge_start_height]
        temp_arc = self.find_tangent_arc(
            start_point.loc['start'],
            bottom_ledge_angle,
            self.arcs['bottom_edge'].points.loc['center'],
            self.arcs['bottom_edge'].radius,
            "right"
            )
        self.arcs['bottom_ledge'] = copy.deepcopy(temp_arc)
        self.arcs['bottom_ledge'].points.loc['start'] = temp_arc.points.loc['end']
        self.arcs['bottom_ledge'].points.loc['end'] = temp_arc.points.loc['start']
        self.arcs['bottom_ledge'].clockwise = not temp_arc.clockwise
        self.arcs['bottom_ledge'].refresh()
        
        start_point = pd.DataFrame(index=['start'],columns=['x','y'],dtype=float)
        start_point.loc['start'] = [face_thickness,0]
        self.arcs['bottom_face'] = self.find_tangent_arc(
            start_point.loc['start'],
            face_angle + np.pi,
            self.arcs['bottom_edge'].points.loc['center'],
            self.arcs['bottom_edge'].radius,
            "left"
            )

        self.arcs['bottom_edge'].points.loc['start'] = self.arcs['bottom_face'].points.loc['end']
        self.arcs['bottom_edge'].points.loc['end'] = self.arcs['bottom_ledge'].points.loc['start']
        self.arcs['bottom_edge'].refresh()
        
        #self.serial = self.generate_serial()
        self.plot()

    def scale(self,scale_factor):
        scaled_hold = copy.deepcopy(self)
        
        for key,this_arc in scaled_hold.arcs.items():
            scaled_hold.arcs[key] = this_arc.scale(scale_factor)

        #scaled_hold.refresh()
        return scaled_hold
    
    def copy(self):
        #not finished, deepcopy doesn't work here
        copied_hold = copy.deepcopy(self)
        
    def reverse(self):
        reversed_hold = copy.deepcopy(self)

        for i,arc in enumerate(self.arcs):
            reversed_hold.arcs.iloc[i] = arc.reverse()
        reversed_hold.arcs = reversed_hold.arcs[::-1]
    
    def plot(self):
        # figure settings
        figure_width = 60 # cm
        figure_height = 100 # cm
        left_right_margin = 10 # cm
        top_bottom_margin = 10 # cm
        
        # Don't change
        left   = left_right_margin / figure_width # Percentage from height
        bottom = top_bottom_margin / figure_height # Percentage from height
        width  = 1 - left*2
        height = 1 - bottom*2
        mm2inch = 1/25.4 # inch per cm
        
        # specifying the width and the height of the box in inches
        fig = plt.figure(figsize=(figure_width*mm2inch,figure_height*mm2inch))
        axes = fig.add_axes((left, bottom, width, height))
        
        # limits settings (important)
        plt.xlim(0, figure_width * width)
        plt.ylim(-(figure_height * height)/2, (figure_height * height)/2)

        for key,this_arc in self.arcs.items():
            this_arc.plot_arc(axes)
            plt.scatter(this_arc.points.loc[:,'x'],this_arc.points.loc[:,'y'])
        
        this_cirlce = plt.Circle((8,16),3.175)
        axes.add_patch(this_cirlce)
        this_cirlce = plt.Circle((8,-16),3.175)
        axes.add_patch(this_cirlce)

        # axes.set_xlim(0,40)
        # axes.set_ylim(-40,40)
        # axes.set_aspect('equal')
        
        plt.show()
        fig.savefig('hold.png', dpi=1000)
        fig.savefig('hold.pdf')
        return fig,axes
        
    #def generate_serial(self):
        #self.serial = f'{self.arcs['top_edge'].points.loc['center']}'

    def find_tangent_arc(self,start_point,start_angle,goal_arc_center,goal_arc_radius,goal_side):
        sx = start_point['x']
        sy = start_point['y']
        gx = goal_arc_center['x']
        gy = goal_arc_center['y']
        gr = goal_arc_radius
        t = start_angle
        c = goal_arc_center - start_point
        d_hat = np.array([np.cos(start_angle),np.sin(start_angle)])
        e_hat = np.array([-np.sin(start_angle),np.cos(start_angle)])
        a = np.dot(c,d_hat)
        b = np.dot(c,e_hat)
        if goal_side == 'left':
            r = (a**2 - gr**2 + b**2)/(2 * (b + gr))
        else:
            r = (a**2 - gr**2 + b**2)/(2 * (b - gr))
        tangent_arc_center =  start_point + r * e_hat
        tangent_point = tangent_arc_center + np.abs(r)*(goal_arc_center-tangent_arc_center)/np.linalg.norm(goal_arc_center - tangent_arc_center)
        
        if r > 0:
            clockwise = False
        else:
            clockwise = True
        
        this_arc = arc(start_point,tangent_point,tangent_arc_center,clockwise)
        print(tangent_arc_center)
        print(r)
        return this_arc
    
    def check_consistency(self):
        previous_arc = []
        #for key,arc in self.arcs.items():
            
    def reverse_profile(self):
        pass
            
def generate_smaller_hold_profile(hold,face_shift):
    # figure out the new top edge position
    ledge_end_vector = hold.arcs['top_ledge'].points.loc[['center','end']].diff().loc['end']
    ledge_start_vector = hold.arcs['top_ledge'].points.loc[['center','start']].diff().loc['start']

    ledge_sweep_angle = np.arcsin(cross2d(ledge_end_vector.values,ledge_start_vector.values)
                                  / (np.linalg.norm(ledge_end_vector) **2)
                                  )

    shift_angle = ledge_sweep_angle*face_shift
    
    ledge_edge_vector = hold.arcs['top_edge'].points.loc['center'] - hold.arcs['top_ledge'].points.loc['center']
    new_top_edge_position = [ledge_edge_vector['x']*np.cos(shift_angle) - ledge_edge_vector['y']*np.sin(shift_angle),
                         ledge_edge_vector['x']*np.sin(shift_angle) + ledge_edge_vector['y']*np.cos(shift_angle)] + hold.arcs['top_ledge'].points.loc['center']
    
    # figure out the new bottom edge position
    ledge_end_vector = hold.arcs['bottom_ledge'].points.loc[['center','end']].diff().loc['end']
    ledge_start_vector = hold.arcs['bottom_ledge'].points.loc[['center','start']].diff().loc['start']

    ledge_sweep_angle = np.arcsin(cross2d(ledge_start_vector.values,ledge_end_vector.values)
                                  / (np.linalg.norm(ledge_end_vector) **2)
                                  )

    shift_angle = ledge_sweep_angle*face_shift
    ledge_edge_vector = hold.arcs['bottom_edge'].points.loc['center'] - hold.arcs['bottom_ledge'].points.loc['center']
    new_bottom_edge_position = [ledge_edge_vector['x']*np.cos(shift_angle) - ledge_edge_vector['y']*np.sin(shift_angle),
                         ledge_edge_vector['x']*np.sin(shift_angle) + ledge_edge_vector['y']*np.cos(shift_angle)] + hold.arcs['bottom_ledge'].points.loc['center']
    
    
    smaller_profile = hold_profile(top_edge_position = new_top_edge_position,
                           top_edge_radius = hold.arcs['top_edge'].radius,
                           top_ledge_angle = hold.arcs['top_ledge'].get_tangent_angle('start'),
                           top_ledge_start_height = hold.arcs['top_ledge'].points.loc['start','y'],
                           bottom_edge_position = new_bottom_edge_position,
                           bottom_edge_radius = hold.arcs['bottom_edge'].radius,
                           bottom_ledge_angle = hold.arcs['bottom_ledge'].get_tangent_angle('end') + np.pi,
                           bottom_ledge_start_height = hold.arcs['bottom_ledge'].points.loc['end','y'],
                           face_angle = hold.arcs['top_face'].get_tangent_angle('end') + np.pi,
                           face_thickness = hold.arcs['top_face'].points.loc['end','x'] * (1- face_shift)
                           )
    return smaller_profile
    #smaller_profile.plot()
def generate_random_hold_profile(seed = -1,hold_height = 40.0,edge_radius = 0,edge_range = [1,3],edge_center = 0,hold_thickness = 0,max_thickness = 38):
    if seed == -1:
        rnd.seed()
    
    #random_hold_profile = hold_profile()
    # Pick the parameters that define the hold
    hold_thickness = rnd.uniform(20,max_thickness)
    top_edge_center_x = rnd.uniform(8,max_thickness-2)
    min_distance_from_edge = max_thickness/2-abs(max_thickness/2-top_edge_center_x)
    top_edge_radius = rnd.uniform(edge_range[0],min_distance_from_edge)
    top_edge_center_y = rnd.uniform(20,hold_height - top_edge_radius)
    top_ledge_start_height = rnd.uniform(max(top_edge_center_y-top_edge_center_x,20),min(top_edge_center_y+top_edge_center_x,hold_height))
    
    top_ledge_angle = rnd.triangular(-np.pi/8,0,np.pi/8)
    bottom_edge_center_x = rnd.uniform(8,max_thickness)
    bottom_edge_radius = rnd.uniform(edge_range[0],max_thickness/2-abs(max_thickness/2-bottom_edge_center_x))
    bottom_edge_center_y = rnd.uniform(-20,-(hold_height - bottom_edge_radius))
    bottom_ledge_angle = rnd.triangular(-np.pi/8,0,np.pi/8)
    bottom_ledge_start_height = rnd.uniform(max(bottom_edge_center_y - bottom_edge_center_x,-hold_height),min(bottom_edge_center_y + bottom_edge_center_x,-20))
    
    random_hold_profile = hold_profile(top_edge_position = [top_edge_center_x,top_edge_center_y],
                       top_edge_radius = top_edge_radius,
                       top_ledge_angle = top_ledge_angle,
                       top_ledge_start_height = top_ledge_start_height,
                       bottom_edge_position = [bottom_edge_center_x,bottom_edge_center_y],
                       bottom_edge_radius = bottom_edge_radius,
                       bottom_ledge_angle = bottom_ledge_angle,
                       face_angle = np.pi/2,
                       bottom_ledge_start_height = bottom_ledge_start_height
                       )

    return random_hold_profile
def generate_profile_arcs_gcode(profile,feedrate,forward=True):
    # assumes cutter compensation is on and that the cutter is positioned at
    # the start of the ledge arc on the profile at the intended Z height
    # with absolute distance mode on
    clockwise_dict = {True:'G2',False:'G3'}
    profile_gcode = []
    if forward:
        for i,a in profile.arcs.items():
            this_arc_gcode = f'{clockwise_dict[a.clockwise]} X{a.points.loc["end","x"]:.4f} Y{a.points.loc["end","y"]:.4f} I{a.points.loc["center","x"]:.4f} J{a.points.loc["center","y"]:.4f}\n'
            profile_gcode = profile_gcode + this_arc_gcode
    else:
        pass
    return profile_gcode,distance

def generate_profile_gcode(profile,z_height,feedrate):
    # assumes cutter compensation is on and that the cutter is positioned at 0,0
    # at some Z > z_height, with absolute distance mode on
    clockwise_dict = {True:'G2',False:'G3'}
    profile_gcode = f'G1 X{profile.arcs.iloc[0].points.loc["start","x"]:.4f} Y{profile.arcs.iloc[0].points.loc["start","y"]:.4f} Z{z_height:.4f} F{feedrate:.4f}\n'
    for i,a in profile.arcs.items():
        this_arc_gcode = f'{clockwise_dict[a.clockwise]} X{a.points.loc["end","x"]:.4f} Y{a.points.loc["end","y"]:.4f} I{a.points.loc["center","x"]:.4f} J{a.points.loc["center","y"]:.4f}\n'
        profile_gcode = profile_gcode + this_arc_gcode
    return profile_gcode

def generate_gcode_3d(hold_series,o_code_number,x_offset,y_offset,x_step):

    #sort the hold profiles so the lowest one gets cut first, but that the highest
    #level of that lowest number hold profile is first
    hold_series = hold_series.sort_values('depth',ignore_index=True,ascending=False)
    last_segment = 9e9
    hold_gcode = 'G90.1'
    for i,row in hold_series.iterrows():
        if row['segment_number'] != last_segment:
            hold_gcode += f'''
G0 Z1.0
G40
G52 X{x_offset - row["segment_number"]*x_step} Y{y_offset}
G42
G0 X0.0 Y0.0 S2000 M3
G1 Z0.9 F10.00
'''
        #profile = row['profile'].scale(1/25.4)
        profile = row['profile']
        hold_profile_gcode = generate_profile_gcode(profile,z_height=row['segment_depth'],feedrate=10)
        hold_gcode += hold_profile_gcode
        last_segment = row['segment_number']
    
    hold_gcode += 'G0 Z1.0\n'
    hold_gcode += 'M5\n'
    hold_gcode += 'M2\n'
    print(hold_gcode)
    with open(r'NC Files\Output.ngc', 'w') as text_file:
        text_file.write(hold_gcode)
    return hold_gcode
    
    
def generate_gcode_2d(hold,o_code_number):
    scaled_hold = hold.scale(1/25.4)
    
    clockwise_dict = {True:'G2',False:'G3'}
    
    preamle = fr'''
o{o_code_number} sub (#1 = z depth-must be negative, #2 = number of steps, #3 = feedrate, #4 = x-offset #5 = y-offset, #6 = rotation)
    G52 X#4 Y#5
    G0 X0.0 Y0.0 S2000 M3
    G0 Z1.0
    G42
    G1 X{scaled_hold.arcs['top_ledge'].points['start','x']:.4f} Y{scaled_hold.arcs['top_ledge'].points['start','y']:.4f} Z0.1 F#3
    #14 = [#1/#2]
    #15 = #14
    o{o_code_number+500} while [#15 GE #1]
'''
    #for key,this_arc in this_hold.arcs.items():
        #this_command = r'{clockwise_dict[this_arc.clockwise]:.4f} X{this_arc.points.loc['end','x']:.4f} Y{this_arc.points.loc['end','y']} I{this_arc.points.loc['center','x']} J{this_arc.points.loc['center','y']}'        
    with open(r'NC Files\Output.ngc', 'w') as text_file:
        text_file.write(
fr'''
o{o_code_number} sub (#1 = z depth-must be negative, #2 = number of steps, #3 = feedrate, #4 = x-offset #5 = y-offset, #6 = rotation)
    G52 X#4 Y#5
    G0 X0.0 Y0.0 S2000 M3
    G0 Z1.0
    G42
    G1 X{arcs_2.loc['ledge','start_x']:.4f} Y{arcs_2.loc['ledge','start_y']:.4f} Z0.1 F#3
    #14 = [#1/#2]
    #15 = #14
    o{o_code_number+500} while [#15 GE #1]
        {concave_edge_2} X{arcs_2.loc['ledge','end_x']:.4f} Y{arcs_2.loc['ledge','end_y']:.4f} I{arcs_2.loc['ledge','center_x']:.4f} J{arcs_2.loc['ledge','center_y']:.4f} Z[#15]
        G3 X{arcs_2.loc['edge','end_x']:.4f} Y{arcs_2.loc['edge','end_y']:.4f} I{arcs_2.loc['edge','center_x']:.4f} J{arcs_2.loc['edge','center_y']:.4f}
        {concave_face_2} X{arcs_2.loc['face','end_x']:.4f} Y{arcs_2.loc['face','end_y']:.4f} I{arcs_2.loc['face','center_x']:.4f} J{arcs_2.loc['face','center_y']:.4f}
        {concave_face_1} X{arcs_1.loc['face','start_x']:.4f} Y{arcs_1.loc['face','start_y']:.4f} I{arcs_1.loc['face','center_x']:.4f} J{arcs_1.loc['face','center_y']:.4f}
        G3 X{arcs_1.loc['edge','start_x']:.4f} Y{arcs_1.loc['edge','start_y']:.4f} I{arcs_1.loc['edge','center_x']:.4f} J{arcs_1.loc['edge','center_y']:.4f}
        {concave_edge_1} X{arcs_1.loc['ledge','start_x']:.4f} Y{arcs_1.loc['ledge','start_y']:.4f} I{arcs_1.loc['ledge','center_x']:.4f} J{arcs_1.loc['ledge','center_y']:.4f}
        G1 X{arcs_2.loc['ledge','start_x']:.4f} Y{arcs_2.loc['ledge','start_y']:.4f}
        #15 = [#15 + #14]
    o{o_code_number+500} endwhile
    Z2.
    G40
    G52 X0 Y0 Z0
o{o_code_number} endsub

o102 call [235] (select tool)
o150 call [-0.75] [3] [20] [0] [0]
o150 call [-0.75] [3] [20] [-1.75] [0]
o150 call [-0.75] [3] [20] [-3.5] [0]
o150 call [-0.75] [3] [20] [-5.25] [0]
m2''')
    return

def save_gcode():
    return

def generate_hold_series(hold_profile,center_width=3/4,width=3/4*2,step=1/16,taper_ratio = 0.2,curve='polynomial'):
    steps = (width - center_width) / step
    coef = taper_ratio / (steps**2)
    hold_profiles = pd.DataFrame(columns=['profile','depth','segment','segment_depth','step_depth'])
    for i,x in enumerate(np.linspace(0,width-step,int(width//step))):
        
        if x <= center_width:
            hold_profiles.loc[i,'profile'] =  copy.deepcopy(hold_profile)
        else:
            steps_after_center = (i - center_width/step)
            hold_profiles.loc[i,'profile'] = generate_smaller_hold_profile(hold_profile,coef * steps_after_center**2)
        hold_profiles.loc[i,'depth'] = x
        hold_profiles.loc[i,'segment_depth'] = (x + 1e-9) % center_width
        hold_profiles.loc[i,'segment_number'] = (x + 1e-9)  // center_width
        hold_profiles.loc[i,'step_depth'] = step
    return hold_profiles
        
        
    
# Function to generate a full hold
# Makes a cadquery shape by extruding the hold profile and cutting the bolt hole away
def generate_hold(o_code_number,seed = -1):
    if seed == -1:
        rnd.seed()
    else:
        rnd.seed(1)
    this_hold = generate_random_hold_profile(seed,hold_height=40,hold_thickness=20).scale(1/25.4)
    #generate_profile_gcode(this_hold, 2,10)
    hold_series = generate_hold_series(this_hold)#,center_width = 6.35,width = 6.35*2)
    
    gcode = generate_gcode_3d(hold_series,o_code_number,0.0,0.0,2.0)
    
    for i,row in hold_series.iterrows():
        this_hold_profile = row['profile']
        if i == 0:
            result = cq.Workplane("right")
        else:
            result = result.copyWorkplane(
                        # create a temporary object with the required workplane
                        cq.Workplane("right", origin=(row['depth'],0,0))
                    )
            
        for key,this_arc in this_hold_profile.arcs.items():
            if key == 'top_ledge':
                result = result.lineTo(this_arc.points.loc['start','x'],this_arc.points.loc['start','y'])
            result = result.threePointArc(this_arc.points.loc['midpoint'].values,this_arc.points.loc['end'].values)
        result = result.close().extrude(row['step_depth'])

    bolt_hole = cq.Workplane("right").polyline([(0,0),(200,0),(200,3/8),(3/4,3/8),(3/4,5),(0,3/16)]).close().revolve(angleDegrees=360, axisStart=(0, 0, 0), axisEnd=(1, 0, 0))
    result = result.cut(bolt_hole)
    mirror = result.mirror(mirrorPlane='YZ')
    result = result.union(mirror)
    
    # result2 = (
    #     cq.Workplane("top")
    #     .center(0.0,100.0)
    #     .lineTo(vec_1.loc['top_corner','x'],vec_1.loc['top_corner','y'])
    #     .threePointArc(arcs_1.loc['ledge',['mid_x','mid_y']].values, arcs_1.loc['ledge',['end_x','end_y']].values)
    #     .threePointArc(arcs_1.loc['edge',['mid_x','mid_y']].values, arcs_1.loc['edge',['end_x','end_y']].values)
    #     .threePointArc(arcs_1.loc['face',['mid_x','mid_y']].values, arcs_1.loc['face',['end_x','end_y']].values)
    #     .threePointArc(arcs_2.loc['face',['mid_x','mid_y']].values, arcs_2.loc['face',['start_x','start_y']].values)
    #     .threePointArc(arcs_2.loc['edge',['mid_x','mid_y']].values, arcs_2.loc['edge',['start_x','start_y']].values)
    #     .threePointArc(arcs_2.loc['ledge',['mid_x','mid_y']].values, arcs_2.loc['ledge',['start_x','start_y']].values)
    #     .lineTo(0.0,-100.0)
    #     .lineTo(100.0,-100.0)
    #     .lineTo(100.0,100.0)
    #     .close()
    #     .extrude(75)
    #     #.translate((-37.5,0,0))
    # )
    
    #result = result2.cut(bolt_hole)
    
    contour_g_code = "0"#generate_gcode(this_hold,o_code_number)
    return result,contour_g_code

# G-Code Preamble String
g_code_contouring_preamble = fr'''
(subroutine to set work offset in terms of machine coords)
	o101 sub
		g10 l2 p1 x0.0 y0.0 z0.0
	o101 ENDsub
    
(setup and tool call - 1st argument is tool number)
	o102 sub
		(set modal codes)
		g00 g17 g40 g80 g90 g90.1 g98 g64 P0.001
		(set work offset)
		o101 call
		(select work offset)
		g54
		
		(stop spindle)
		m5
		(first optional stop)
		m1
		(rapid to tool change position)
		g0 g53 z0 
		g53 x0 y-10
		(select tool)
		t #1 m6
		(apply tool offset)
		g43 h #1
		(second optional stop)
		m1
	o102 ENDsub
'''

# Generate a whole bunch of holds and put them into an array
#shapes_i = []
#test = cq.Workplane()
holds_to_generate = 1
holds_generated = 0
test = cq.Assembly()
contour_g_codes = []
# Generate 10 STEP files
for i in range(holds_to_generate):
    print(i)
    #if i = 0:
    #    test = create_revolved_shape(this_radius_seed=i)  # Use i as seed for randomness
    #else
    for j in range(holds_to_generate):
        print(j)
        #try:
        shape,this_contour_g_code = generate_hold(o_code_number=(200+holds_generated),seed=1)  # Use i as seed for randomness
        contour_g_codes.append(this_contour_g_code)
        test.add(shape, loc=cq.Location(cq.Vector(150.0*i, 150.0*j, 0.0),(0,0,1),rnd.randint(0, 180)))
        #shapes.append(shape)
        file_name = f"revolved_shape_{i}_{j}.step"
        #shape.save('f"hold_{i}_{j}.step"')
        cq.exporters.export(shape, file_name)
        print(f"Generated {file_name}")
        holds_generated += 1
        # except Exception as e:
        #     print(e)
        #     continue

# test = (cq.Assembly(shapes[0], cq.Location(cq.Vector(0, 0, 0)), name="root")
#         #.add(shapes[0], loc=cq.Location(cq.Vector(0, 0, 6)))
#         )
# for i in range(holds_generated-1):
#     test.add(shapes[i+1], loc=cq.Location(cq.Vector(100*(i+1), 0, 0)))

# test = (cq.Workplane().union(shapes[1].translate([6,0,0]))
#         .add(shapes[1],loc=(6,0,0))
#         .add(shapes[2],loc=(18,0,0))
#         )

test.save('assembly.step')
