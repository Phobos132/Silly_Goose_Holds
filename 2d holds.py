import cadquery as cq
import random as rnd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import copy
import sg_holds as sg
#import sympy as sym

# This script generates a 2d profile climbing hold that can be cut from
# a 2x4


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
    
    random_hold_profile = sg.profile(top_edge_position = [top_edge_center_x,top_edge_center_y],
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

    
def generate_profile_arcs_gcode(profile,z_height,feedrate,forward=True,ramp=True):
    # assumes cutter compensation is on and that the cutter is positioned at
    # the start of the ledge arc on the profile at the intended Z height
    # with absolute distance mode on
    clockwise_dict = {True:'G2',False:'G3'}
    distance = 0
    
    if forward:
        this_profile = profile
    else:
        this_profile = profile.reverse()
    
    if ramp:
        ramp_height = 0.1
        
    else:
        ramp_height = 0.0
        
    profile_gcode = f'G1 X{profile.arcs.iloc[0].points.loc["start","x"]:.4f} Y{profile.arcs.iloc[0].points.loc["start","y"]:.4f} Z{z_height + ramp_height:.4f} F{feedrate:.4f}\n'
    
    for i,a in this_profile.arcs.items():
        if i == 'bottom_ledge':
            this_arc_gcode = f'{clockwise_dict[a.clockwise]} X{a.points.loc["end","x"]:.4f} Y{a.points.loc["end","y"]:.4f} I{a.points.loc["center","x"]:.4f} J{a.points.loc["center","y"]:.4f} Z{z_height + ramp_height:.4f}\n'
        else:
            this_arc_gcode = f'{clockwise_dict[a.clockwise]} X{a.points.loc["end","x"]:.4f} Y{a.points.loc["end","y"]:.4f} I{a.points.loc["center","x"]:.4f} J{a.points.loc["center","y"]:.4f} Z{z_height:.4f}\n'
        profile_gcode = profile_gcode + this_arc_gcode
        distance += a.get_arc_length()
    
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


def generate_gcode_3d(hold_series,o_code_number,x_offset,y_offset,rotation_rad):

    hold_series = hold_series.sort_values('depth',ignore_index=True,ascending=False)
    hold_gcode = g_code_contouring_preamble + f'''
O101 call
O102 call [239]
G0 X0 Y1.0 Z{hold_series.iloc[0]['depth'] + 2.0}
G40
(G52 X{x_offset} Y{y_offset})
G41
G0 X0.0 Y0.0 S2000 M3
G1 Z{hold_series.iloc[0]['depth']+0.1} F40.00
'''
    for i,row in hold_series.iterrows():
        profile = row['profile'].scale(1/25.4).rotate(rotation_rad)
        #profile.plot()
        #row['profile'].plot()
        #profile = row['profile']
        hold_profile_gcode,distance = generate_profile_arcs_gcode(profile,z_height=row['depth'],feedrate=40)
        hold_gcode += hold_profile_gcode
    
    hold_gcode += f'G0 Z{hold_series.iloc[0]["depth"] + 2.0}\n'
    hold_gcode += 'M5\n'
    hold_gcode += 'M2\n'
    print(hold_gcode)
    with open(r'NC Files\Output.ngc', 'w') as text_file:
        text_file.write(hold_gcode)
    return hold_gcode

def generate_gcode_3d_sliced(hold_series,o_code_number,x_offset,y_offset,x_step):

    hold_series = hold_series.sort_values('depth',ignore_index=True,ascending=False)
    last_segment = 9e9
    hold_gcode = 'G90.1'
    for i,row in hold_series.iterrows():
        if row['segment_number'] != last_segment:
            hold_gcode += f'''
G0 Z6.0
G40
G52 X{x_offset - row["segment_number"]*x_step} Y{y_offset}
G42
G0 X0.0 Y0.0 S2000 M3
G1 Z0.9 F10.00
'''
        profile = row['profile'].scale(1/25.4)
        #profile.plot()
        #row['profile'].plot()
        #profile = row['profile']
        hold_profile_gcode,distance = generate_profile_arcs_gcode(profile,z_height=row['segment_depth'],feedrate=10,forward=(i%2 == 0))
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

def generate_hold_series(hold_profile,center_width=0,width=1.625,step=1/16,taper_ratio = 0.5,curve='polynomial'):
    steps = (width - center_width) / step
    coef = taper_ratio / (steps**2)
    hold_profiles = pd.DataFrame(columns=['profile','depth','segment','segment_depth','step_depth'])
    for i,x in enumerate(np.linspace(0,width-step,int(width//step))):
        
        if x <= center_width:
            hold_profiles.loc[i,'profile'] =  copy.deepcopy(hold_profile)
        else:
            steps_after_center = (i - center_width/step)
            hold_profiles.loc[i,'profile'] = sg.generate_smaller_profile(hold_profile,coef * steps_after_center**2)
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
    this_hold = generate_random_hold_profile(seed,hold_height=40,hold_thickness=20)
    flipped_hold = this_hold.flip()
    flipped_hold.plot()
    this_hold.plot()
    #generate_profile_gcode(this_hold, 2,10)
    hold_series = generate_hold_series(flipped_hold)#,center_width = 6.35,width = 6.35*2)
    
    gcode = generate_gcode_3d(hold_series,o_code_number,0.0,0.0,-np.pi/2)
    
    for i,row in hold_series.iterrows():
        this_hold_profile = row['profile'].scale(1/25.4)
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
    #result = result.cut(bolt_hole)
    mirror = result.mirror(mirrorPlane='YZ')
    result = result.union(mirror)
    
    
    contour_g_code = "0"#generate_gcode(this_hold,o_code_number)
    return result,gcode


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
        shape,this_contour_g_code = generate_hold(o_code_number=(200+holds_generated),seed=2)  # Use i as seed for randomness
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
