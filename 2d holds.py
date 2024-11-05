import cadquery as cq
import random as rnd
import numpy as np
import pandas as pd
import sympy as sym

# This script generates a 2d profile climbing hold that can be cut from
# a 2x4

#TODO:
    #make it so the ledge arc cant go into negative x
    #if the edge is too close to the origin and the face and ledge
    #radii are two large you get problems
def create_half_hold(seed = -1,hold_height = 100.0,edge_radius = 0,edge_range = [1,20,3],edge_center = 0,hold_thickness = 0):
    if seed == -1:
        rnd.seed()
    
    # Pick the parameters that define the hold
    if edge_radius <= 0:
        edge_radius = rnd.triangular(edge_range[0],edge_range[1],edge_range[2])
    
    # Calculate the tangent arc of the front of the hold
    if hold_thickness <= 0:
        hold_thickness = rnd.uniform(20,38)
    
    # Pick the horizontal center of the edge, at least 
    # one radius away from both edges of the 2x4
    vec = pd.DataFrame(columns=['x','y'])
    if edge_center <= 0:
        vec.loc['edge_center','x'] = rnd.uniform(edge_radius,38-edge_radius)
    
    # Pick the vertical center of the edge, ensure the
    lowest_edge_center = max(-(hold_height/2.0-25),-(vec.loc['edge_center','x'] + edge_radius))
    highest_edge_center = min((hold_height/2.0-25-edge_radius),vec.loc['edge_center','x'] - edge_radius)
    if highest_edge_center < lowest_edge_center:
        raise Exception("edge_radius too big for hold height")
    vec.loc['edge_center','y'] = rnd.uniform(
        lowest_edge_center,highest_edge_center)
    
    # Pick a radius that is not too small, adjust the lognorm values to tweak
    # things to a higher or lower radius or change the variance
    ledge_radius = (rnd.lognormvariate(0, 1) 
                    +  np.linalg.norm(vec.loc['edge_center'], axis=0)
                    )
    
    # Chose the top corner of the hold where it meets the wall, at the start
    # this is 0,0
    vec.loc['top_corner'] = [0,0]
    
    # Start calculating the tangent arc of the ledge
    # this is done using the cosine law
    horizontal_to_edge_center_angle = np.arctan(vec.loc['edge_center','y']/vec.loc['edge_center','x'])
    a = np.linalg.norm(vec.loc['edge_center'], axis=0)
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
    
    vec.loc['ledge_center'] = [ledge_radius*np.cos(horizontal_to_ledge_center_angle),
                               ledge_radius*np.sin(horizontal_to_ledge_center_angle)]   
    
    # Offset everything vertically so the highest point is at the top edge of the
    # blank.
    if concave:
        zero_offset = max(0,vec.loc['edge_center','y'] + edge_radius)
    else:
        zero_offset = max(0,vec.loc['edge_center','y'] + edge_radius,
                            vec.loc['ledge_center','y'] + ledge_radius)
    vec['y'] = vec['y'] - zero_offset + hold_height/2
    
    face_offset = hold_thickness - vec.loc['edge_center','x']
    # Note that a negative radius here indicates that the suface is convex
    face_radius = ((edge_radius**2 - face_offset**2 - vec.loc['edge_center','y']**2)
                   /(2*(face_offset-edge_radius))
                   )
    vec.loc['face_center'] = [hold_thickness+face_radius,0]
    
    #Now that we have the location of the tangent circles that will make our
    # hold, generate the actual arcs:
    arcs = pd.DataFrame(columns=['start_x','start_y','end_x','end_y','mid_x','mid_y','radius'])
    arcs.loc['ledge','radius'] = ledge_radius
    arcs.loc['edge','radius'] = edge_radius
    arcs.loc['face','radius'] = abs(face_radius)
    
    # Find the start of the ledge arc
    arcs.loc['ledge',['start_x','start_y']] = vec.loc['top_corner'].values
    
    # Find the end of the ledge arc
    r = vec.loc['edge_center']-vec.loc['ledge_center']
    arcs.loc['ledge',['end_x','end_y']] = ( vec.loc['ledge_center'] 
                                               + r/np.linalg.norm(r)*ledge_radius
                                               ).values
    
    # Find a point that is around the middle of the ledge arc
    r = vec.loc[['edge_center','top_corner']].mean() - vec.loc['ledge_center']
    arcs.loc['ledge',['mid_x','mid_y']] = ( vec.loc['ledge_center'] 
                                           + r/np.linalg.norm(r)*ledge_radius
                                           ).values
    
    # Find the end of the face arc
    arcs.loc['face',['end_x','end_y']] = [hold_thickness,0]
    
    # Find the start of the face arc
    r = vec.loc['edge_center']-vec.loc['face_center']
    arcs.loc['face',['start_x','start_y']] = ( vec.loc['face_center'] 
                                               + r/np.linalg.norm(r)*abs(face_radius)
                                               ).values
    
    # Check that the face and ledge arcs don't reach around and overlap
    if arcs.loc['face','start_x'] < arcs.loc['ledge','end_x']:
        return pd.DataFrame(),pd.DataFrame()
    
    # Find a point that is around the middle of the face arc
    r = vec.loc['edge_center']/2 - vec.loc['face_center']
    arcs.loc['face',['mid_x','mid_y']] = ( vec.loc['face_center'] 
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
    arcs.loc['edge',['mid_x','mid_y']] = ( vec.loc['edge_center'] 
                                           + r/np.linalg.norm(r)*edge_radius
                                           ).values
    if arcs.loc['edge',['mid_x','mid_y']].sum() < vec.loc['edge_center'].sum():
        arcs.loc['edge',['mid_x','mid_y']] = ( vec.loc['edge_center'] 
                                               - r/np.linalg.norm(r)*edge_radius
                                               ).values
    return arcs,vec

def generate_hold(seed = -1):
    if seed == -1:
        rnd.seed()
    
    arcs_1 = pd.DataFrame()
    while arcs_1.empty:
        arcs_1,vec_1 = create_half_hold(seed = rnd.randint(0, 99999))
    
    arcs_2 = pd.DataFrame()
    while arcs_2.empty:
        arcs_2,vec_2 = create_half_hold(seed = rnd.randint(0, 99999),hold_thickness = arcs_1.loc['face','end_x'])
    
    arcs_2[['start_y','mid_y','end_y']] = -arcs_2[['start_y','mid_y','end_y']]
    vec_2['y'] = -vec_2['y']
    
    result = (
        cq.Workplane("right")
        .lineTo(vec_1.loc['top_corner','x'],vec_1.loc['top_corner','y'])
        .threePointArc(arcs_1.loc['ledge',['mid_x','mid_y']].values, arcs_1.loc['ledge',['end_x','end_y']].values)
        .threePointArc(arcs_1.loc['edge',['mid_x','mid_y']].values, arcs_1.loc['edge',['end_x','end_y']].values)
        .threePointArc(arcs_1.loc['face',['mid_x','mid_y']].values, arcs_1.loc['face',['end_x','end_y']].values)
        .threePointArc(arcs_2.loc['face',['mid_x','mid_y']].values, arcs_2.loc['face',['start_x','start_y']].values)
        .threePointArc(arcs_2.loc['edge',['mid_x','mid_y']].values, arcs_2.loc['edge',['start_x','start_y']].values)
        .threePointArc(arcs_2.loc['ledge',['mid_x','mid_y']].values, arcs_2.loc['ledge',['start_x','start_y']].values)
        .close()
        .extrude(75)
    )
    
    bolt_hole = cq.Workplane("right").polyline([(0,0),(200,0),(200,10),(15,10),(15,5),(0,5)]).close().revolve(angleDegrees=360, axisStart=(0, 0, 0), axisEnd=(1, 0, 0)).translate((37.5,0,0))
    result = result.cut(bolt_hole)
    
    return result


shapes_i = []
test = cq.Workplane()
holds_to_generate = 1
holds_generated = 0
test = cq.Assembly()
# Generate 10 STEP files
for i in range(holds_to_generate):
    #if i = 0:
    #    test = create_revolved_shape(this_radius_seed=i)  # Use i as seed for randomness
    #else
    for j in range(holds_to_generate):
        try:
            shape = generate_hold(seed=i)  # Use i as seed for randomness
            test.add(shape, loc=cq.Location(cq.Vector(150.0*i, 150.0*j, 0.0),(0,0,1),rnd.randint(0, 180)))
            shapes.append(shape)
            test = test.union(shape.translate([6*i,0,0]))
            file_name = f"revolved_shape_{i}.step"
            cq.exporters.export(shape, file_name)
            print(f"Generated {file_name}")
            holds_generated += 1
        except:
            continue

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