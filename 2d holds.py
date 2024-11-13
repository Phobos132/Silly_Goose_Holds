import cadquery as cq
import random as rnd
import numpy as np
import pandas as pd
import sympy as sym

# This script generates a 2d profile climbing hold that can be cut from
# a 2x4
class arc:
    points = pd.DataFrame(index=['start','end','center'],columns=['x','y'],dtype=float)
    clockwise = False
    radius = 0.0
    def __init__(self,start_point = [0,0]
                 ,end_point = [1,1],
                 center_point = [1,0],
                 clockwise_in = True):
        self.points.loc['start'] = start_point
        self.points.loc['end'] = end_point
        self.points.loc['center'] = center_point
        self.clockwise = clockwise_in
        self.check_radius()
        
    def check_radius(self):
        if (np.linalg.norm(self.points.loc['start'] 
                           - self.points.loc['center'])
            != np.linalg.norm(self.points.loc['end'] 
                              - self.points.loc['center'])):
            raise Exception("This is from the 'arc' class: the distance from "
                            "the center to the end and the center to the start"
                            " are different!")
        else:
            self.radius = np.linalg.norm(self.points.loc['start'] 
                                          - self.points.loc['center'])

my_arc = arc([0,0],[2,2],[2,0],True)
my_arc.points.loc['start','x'] = 0
my_arc.clockwise
my_arc.radius

class hold:
    def __init__(self,
                 top_edge_position = [30,30],
                 top_edge_radius = 10,
                 top_ledge_angle = 0,
                 top_ledge_start_height = 30,
                 bottom_edge_position = [30,-30],
                 bottom_edge_radius = 10,
                 bottom_ledge_angle = np.pi/2,
                 bottom_ledge_start_height = -30,
                 face_angle = np.pi/2
                 ):
        top_edge = arc()
        top_edge.points.loc['center'] = top_edge_position
        top_edge.radius = top_edge_radius
        top_ledge = arc()
        top_ledge.points.loc['start'] = [0,top_ledge_start_height]
        self.find_tangent_arc(top_ledge.points.loc['start'],
                         top_ledge_angle,
                         top_edge.points.loc['center'],
                         top_edge.radius)
        top_face = arc()
        bottom_ledge = arc()
        bottom_edge = arc()
        bottom_face = arc()
        
    def find_tangent_arc(self,start_point,start_angle,goal_arc_center,goal_arc_radius):
        sx = start_point['x']
        sy = start_point['y']
        gx = goal_arc_center['x']
        gy = goal_arc_center['y']
        gr = goal_arc_radius
        t = start_angle
        b = (np.tan(t)*(gy-sy) + gx - sx) / (np.tan(t)*np.sin(t) + np.cos(t))
        a = (gx - sx - b*np.cos(t)) / np.sin(t)
        r = (a**2 - b**2 - gr**2) / (2*gr+2*a)
        tangent_arc_center = [r*np.cos(t-np.pi/2),r*np.cos(t-np.pi/2)]
        return tangent_arc_center,r

test_hold = hold()
#TODO:
    #make it so the ledge arc cant go into negative x
    #if the edge is too close to the origin and the face and ledge
    #radii are two large you get problems
def create_half_hold(seed = -1,hold_height = 60.0,edge_radius = 0,edge_range = [1,20,3],edge_center = 0,hold_thickness = 0):
    if seed == -1:
        rnd.seed()
    
    # Pick the parameters that define the hold
    if edge_radius <= 0:
        edge_radius = rnd.triangular(edge_range[0],edge_range[1],edge_range[2])
        
    if hold_thickness <= 0:
        hold_thickness = rnd.uniform(20,38)
    
    # Pick the horizontal center of the edge, at least 
    # one radius away from both edges of the 2x4
    vec = pd.DataFrame(columns=['x','y'])
    arcs = pd.DataFrame(columns=['start_x','start_y','end_x','end_y','mid_x','mid_y','center_x','center_y','radius','concave'])
    if edge_center <= 0:
        #vec.loc['edge_center','x'] = rnd.uniform(max(edge_radius,8),min(38-edge_radius,edge_radius*8))
        arcs.loc['edge','center_x'] = rnd.uniform(max(edge_radius,8),min(38-edge_radius,edge_radius*8))
    
    #if hold_thickness < vec.loc['edge_center','x'] + edge_radius:
    #    hold_thickness = vec.loc['edge_center','x'] + edge_radius + 1
    
    # Pick the vertical center of the edge
    lowest_edge_center = max(-(hold_height/2.0-25),-(arcs.loc['edge','center_x'] + edge_radius))
    highest_edge_center = min((hold_height/2.0-25-edge_radius),arcs.loc['edge','center_x'] - edge_radius)
    if highest_edge_center < lowest_edge_center:
        return pd.DataFrame(),pd.DataFrame(),0
        #raise Exception("edge_radius too big for hold height")
    #vec.loc['edge_center','y'] = rnd.uniform(lowest_edge_center,highest_edge_center)
    arcs.loc['edge','center_y'] = rnd.uniform(lowest_edge_center,highest_edge_center)
    
    # Pick a ledge radius that is not too small, adjust the lognorm values to tweak
    # things to a higher or lower radius or change the variance
    ledge_radius = (rnd.lognormvariate(0, 1) 
                    +  np.linalg.norm(arcs.loc['edge',['center_x','center_y']], axis=0)
                    )
    
    # Chose the top corner of the hold where it meets the wall, at the start
    # this is 0,0
    #vec.loc['top_corner'] = [0,0]
    arcs.loc['ledge',['start_x','start_y']] = [0,0]
    
    # Start calculating the tangent arc of the ledge
    # this is done using the cosine law
        #horizontal_to_edge_center_angle = np.arctan(vec.loc['edge_center','y']/vec.loc['edge_center','x'])
    horizontal_to_edge_center_angle = np.arctan(arcs.loc['edge','center_y']/arcs.loc['edge','center_x'])
    a = np.linalg.norm(arcs.loc['edge',['center_x','center_y']], axis=0)
    b = ledge_radius
    # Decide if it should be concave or convex
    concave = rnd.randint(0,1)
    if concave:
        c = ledge_radius + edge_radius
        edge_center_to_ledge_center_angle = np.arccos((c**2 - a**2 - b**2)/(-2*a*b))
        horizontal_to_ledge_center_angle = ( horizontal_to_edge_center_angle 
                                            + edge_center_to_ledge_center_angle)
    else:
        c = ledge_radius - edge_radius
        edge_center_to_ledge_center_angle = np.arccos((c**2 - a**2 - b**2)/(-2*a*b))
        horizontal_to_ledge_center_angle = ( horizontal_to_edge_center_angle 
                                            - edge_center_to_ledge_center_angle)
    #vec.loc['ledge_center'] = [ledge_radius*np.cos(horizontal_to_ledge_center_angle),ledge_radius*np.sin(horizontal_to_ledge_center_angle)]   
    arcs.loc['ledge',['center_x','center_y']] = [ledge_radius*np.cos(horizontal_to_ledge_center_angle),ledge_radius*np.sin(horizontal_to_ledge_center_angle)]   
    
    # Offset everything vertically so the highest point is at the top edge of the
    # blank.
    if concave:
        #zero_offset = max(0,vec.loc['edge_center','y'] + edge_radius)
        zero_offset = max(0,arcs.loc['ledge','center_y'] + edge_radius)
    else:
        #zero_offset = max(0,vec.loc['edge_center','y'] + edge_radius,vec.loc['ledge_center','y'] + ledge_radius)
        zero_offset = max(0,arcs.loc['ledge','center_y'] + edge_radius,arcs.loc['ledge','center_y'] + ledge_radius)
    #vec['y'] = vec['y'] - zero_offset + hold_height/2
    arcs[arcs.filter(like='_y').columns] = arcs.filter(like='_y') - zero_offset + hold_height/2

    # Start calculating the arc that forms the face of the hold
    face_offset = hold_thickness - arcs.loc['edge','center_x']
    # Note that a negative radius here indicates that the suface is convex
    face_radius = ((edge_radius**2 - face_offset**2 - arcs.loc['edge','center_y']**2)
                   /(2*(face_offset-edge_radius))
                   )
    #vec.loc['face_center'] = [hold_thickness+face_radius,0]
    arcs.loc['face',['center_x','center_y']] = [hold_thickness+face_radius,0]
    
    #Now that we have the location of the tangent circles that will make our
    # hold, generate the actual arcs:
    arcs.loc['ledge','radius'] = ledge_radius
    arcs.loc['edge','radius'] = edge_radius
    arcs.loc['face','radius'] = face_radius

    #arcs.loc['ledge',['center_x','center_y']] = vec.loc['ledge_center'].values
    #arcs.loc['edge',['center_x','center_y']] = vec.loc['edge_center'].values
    #arcs.loc['face',['center_x','center_y']] = vec.loc['face_center'].values

    # Find the start of the ledge arc
    #arcs.loc['ledge',['start_x','start_y']] = vec.loc['top_corner'].values
    
    # Find the end of the ledge arc
    r = arcs.loc['edge',['center_x','center_y']] - arcs.loc['ledge',['center_x','center_y']]
    arcs.loc['ledge',['end_x','end_y']] = ( arcs.loc['ledge',['center_x','center_y']] 
                                               + r/np.linalg.norm(r)*ledge_radius
                                               ).values
    
    # Find a point that is around the middle of the ledge arc
    r = (arcs.loc['ledge',['end_x','end_y']] + arcs.loc['ledge',['end_x','end_y']])/2  - arcs.loc['ledge',['center_x','center_y']].values
    arcs.loc['ledge',['mid_x','mid_y']] = ( arcs.loc['ledge',['center_x','center_y']].values
                                           + r/np.linalg.norm(r)*ledge_radius
                                           ).values
    
    # Find the end of the face arc
    arcs.loc['face',['end_x','end_y']] = [hold_thickness,0]
    
    # Find the start of the face arc
    r = arcs.loc['edge',['center_x','center_y']]-arcs.loc['face',['center_x','center_y']]
    arcs.loc['face',['start_x','start_y']] = ( arcs.loc['face',['center_x','center_y']]
                                               + r/np.linalg.norm(r)*abs(face_radius)
                                               ).values
    
    # Check that the face and ledge arcs don't reach around and overlap
    if arcs.loc['face','start_x'] < arcs.loc['ledge','end_x']:
        return pd.DataFrame(),pd.DataFrame(),0
    
    # Find a point that is around the middle of the face arc
    r = (arcs.loc['face',['start_x','start_y']].values + arcs.loc['face',['end_x','end_y']].values)/2 - arcs.loc['face',['center_x','center_y']]
    arcs.loc['face',['mid_x','mid_y']] = ( arcs.loc['face',['center_x','center_y']]
                                           + r/np.linalg.norm(r)*abs(face_radius)
                                           ).values
    
    # The start of the edge arc is the end of the ledge arc
    arcs.loc['edge',['start_x','start_y']] = arcs.loc['ledge',['end_x','end_y']].values
    
    # The end of the edge arc is the start of the face arc
    arcs.loc['edge',['end_x','end_y']] = arcs.loc['face',['start_x','start_y']].values
    
    # Find a point around the middle of the edge arc
    # This doesn't work if the arc is more than 180 degrees, should do it with
    # angles instead
    r = (arcs.loc['edge',['start_x','start_y']].values + arcs.loc['edge',['end_x','end_y']].values)/2 - vec.loc['edge_center'] 
    arcs.loc['edge',['mid_x','mid_y']] = ( arcs.loc['edge',['center_x','center_y']] 
                                           + r/np.linalg.norm(r)*edge_radius
                                           ).values
    if arcs.loc['edge',['mid_x','mid_y']].sum() < arcs.loc['edge',['center_x','center_y']].sum():
        arcs.loc['edge',['mid_x','mid_y']] = ( arcs.loc['face',['center_x','center_y']] 
                                               - r/np.linalg.norm(r)*edge_radius
                                               ).values
    return arcs,vec,concave

def generate_gcode(arcs_1,arcs_2,concave_1,concave_2,o_code_number):
    arcs_1 = arcs_1/25.4
    arcs_2 = arcs_2/25.4
    
    if concave_1:
        concave_edge_1 = 'G2'
    else:
        concave_edge_1 = 'G3'
        
    if concave_2:
        concave_edge_2 = 'G2'
    else:
        concave_edge_2 = 'G3'
        
    if arcs_1.loc['face','radius'] > 0:
        concave_face_1 = 'G2'
    else:
        concave_face_1 = 'G3'
        
    if arcs_2.loc['face','radius'] > 0:
        concave_face_2 = 'G2'
    else:
        concave_face_2 = 'G3'
        
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

# Function to generate a full hold
# Generates two half hold profiles with the same thickness, mirrors one vertically then sticks them together
# Makes a cadquery shape by extruding the hold profile and cutting the bolt hole away
def generate_hold(o_code_number,seed = -1,):
    if seed == -1:
        rnd.seed()
    
    arcs_1 = pd.DataFrame()
    while arcs_1.empty:
        arcs_1,vec_1,concave_1 = create_half_hold(seed = rnd.randint(0, 99999))
    
    arcs_2 = pd.DataFrame()
    while arcs_2.empty:
        arcs_2,vec_2,concave_2 = create_half_hold(seed = rnd.randint(0, 99999),hold_thickness = arcs_1.loc['face','end_x'])
    
    arcs_2[['start_y','mid_y','end_y','center_y']] = -arcs_2[['start_y','mid_y','end_y','center_y']]
    vec_2['y'] = -vec_2['y']
    
    result = (
        cq.Workplane("right")
        .lineTo(arcs_1.loc['ledge',['start_x','start_y']].values,vec_1.loc['top_corner','y'])
        .threePointArc(arcs_1.loc['ledge',['mid_x','mid_y']].values, arcs_1.loc['ledge',['end_x','end_y']].values)
        .threePointArc(arcs_1.loc['edge',['mid_x','mid_y']].values, arcs_1.loc['edge',['end_x','end_y']].values)
        .threePointArc(arcs_1.loc['face',['mid_x','mid_y']].values, arcs_1.loc['face',['end_x','end_y']].values)
        .threePointArc(arcs_2.loc['face',['mid_x','mid_y']].values, arcs_2.loc['face',['start_x','start_y']].values)
        .threePointArc(arcs_2.loc['edge',['mid_x','mid_y']].values, arcs_2.loc['edge',['start_x','start_y']].values)
        .threePointArc(arcs_2.loc['ledge',['mid_x','mid_y']].values, arcs_2.loc['ledge',['start_x','start_y']].values)
        .close()
        .extrude(75)
        .translate((-37.5,0,0))
    )
    
    bolt_hole = cq.Workplane("right").polyline([(0,0),(200,0),(200,10),(18,10),(18,5),(0,5)]).close().revolve(angleDegrees=360, axisStart=(0, 0, 0), axisEnd=(1, 0, 0))
    result = result.cut(bolt_hole)
    
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
    
    contour_g_code = generate_gcode(arcs_1, arcs_2, concave_1, concave_2,o_code_number)
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
        shape,this_contour_g_code = generate_hold(o_code_number=(200+holds_generated))  # Use i as seed for randomness
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
