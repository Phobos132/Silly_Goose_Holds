import cadquery as cq
import random

# Function to create a revolved shape with a random profile this_radius
def create_revolved_shape(this_radius_seed):
    random.seed(this_radius_seed)

    
    # Create a valid profile: a vertical line and an arc that closes the wire
    # profile = (
    #     cq.Workplane("XY")
    #     .moveTo(0, 0)               # Start at the origin
    #     .lineTo(0, 20)              # Vertical line
    #     .this_radiusArc((this_radius, 0), -this_radius)  # Arc to make the profile closed
    #     .close()
    # )
    
    #s = cq.Workplane("XY")
    
    inverse_block_1 = cq.Workplane("front").rect(200, 200).extrude(200)
    inverse_block_2 = cq.Workplane("front").rect(200, 200).rect(3.5, 8).extrude(-200).rect(200, 200).rect(3.5, 8).extrude(200)
    bolt_hole = cq.Workplane("right").polyline([(0,0),(200,0),(200,0.25),(0.25,0.25),(0.25,0.125),(0,0.125)]).close().revolve(angleDegrees=360, axisStart=(0, 0, 0), axisEnd=(1, 0, 0))
    
    wall_width = random.uniform(1, 2)
    top_rise = random.uniform(-1, 1)
    top_depth = random.uniform(0.5, 1.5)
    bottom_rise = random.uniform(-1, 1)
    bottom_depth = random.uniform(0.5, 1.5)
    this_radius = random.triangular(0.5, 20,5)
    fillet_radius = random.triangular(0.05, 0.5,0.1)
    
    
    sPnts1 = [
        (-this_radius, wall_width),
        (0, wall_width),
        (top_depth, wall_width+top_rise),
        (bottom_depth, -wall_width-bottom_rise),
        (0, -wall_width),
        (-this_radius, -wall_width),
    ]
    
    result = cq.Workplane("right").polyline(sPnts1).close()
   
    revolved_shape = result.revolve(angleDegrees=360, axisStart=(-this_radius, 0, 0), axisEnd=(-this_radius, 1, 0)).fillet(fillet_radius)
    
    revolved_shape = revolved_shape.cut(inverse_block_2).edges().fillet(0.25)
    revolved_shape = revolved_shape.cut(inverse_block_1)
    revolved_shape = revolved_shape.cut(bolt_hole)

    
    return revolved_shape

shapes = []
test = cq.Workplane()
holds_to_generate = 10
holds_generated = 0
# Generate 10 STEP files
for i in range(holds_to_generate):
    #if i = 0:
    #    test = create_revolved_shape(this_radius_seed=i)  # Use i as seed for randomness
    #else
    try:
        shape = create_revolved_shape(this_radius_seed=i)  # Use i as seed for randomness
        shapes.append(shape)
        test = test.union(shape.translate([6*i,0,0]))
        file_name = f"revolved_shape_{i+1}.step"
        cq.exporters.export(shape, file_name)
        print(f"Generated {file_name}")
        holds_generated += 1
    except:
        continue

test = (cq.Assembly(shapes[0], cq.Location(cq.Vector(0, 0, 0)), name="root")
        #.add(shapes[0], loc=cq.Location(cq.Vector(0, 0, 6)))
        )
for i in range(holds_generated-1):
    test.add(shapes[i+1], loc=cq.Location(cq.Vector(6*(i+1), 0, 0)))

# test = (cq.Workplane().union(shapes[1].translate([6,0,0]))
#         .add(shapes[1],loc=(6,0,0))
#         .add(shapes[2],loc=(18,0,0))
#         )

test.save('assembly.step')
#cq.exporters.export(test, 'assembly.stl')
#cq.exporters.export(test, 'assembly.step')
