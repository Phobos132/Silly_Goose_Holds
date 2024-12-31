#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 10:44:07 2024

@author: evan
"""
#import cadquery as cq
import random as rnd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import copy

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
    
    def plot_arc(self,axes,color='black'):
        start_vector = self.points.loc['start'] - self.points.loc['center']
        start_angle = np.arctan2(start_vector['y'],start_vector['x'])*180.0/np.pi % 360.0
        end_vector = self.points.loc['end'] - self.points.loc['center']
        end_angle = np.arctan2(end_vector['y'],end_vector['x'])*180.0/np.pi % 360.0
        if self.clockwise:
            arc = patch.Arc(self.points.loc['center'],self.radius*2,self.radius*2, theta2 = start_angle, theta1 = end_angle, edgecolor=color)
        else:
            arc = patch.Arc(self.points.loc['center'],self.radius*2,self.radius*2, theta1 = start_angle, theta2 = end_angle, edgecolor=color)
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
    
    def rotate(self,rotation_angle_rad):

        rot_matrix = np.array([[np.cos(rotation_angle_rad), np.sin(rotation_angle_rad)], 
                                  [-np.sin(rotation_angle_rad),  np.cos(rotation_angle_rad)]])

        rotated_arc = arc(start_point = np.matmul(self.points.loc['start'], rot_matrix),
                           end_point = np.matmul(self.points.loc['end'], rot_matrix),
                           center_point = np.matmul(self.points.loc['center'], rot_matrix),
                           clockwise_in = self.clockwise)
        return rotated_arc
    
    def flip_vertical(self):
        flipped_arc = self.copy()
        flipped_arc.points.loc[:,'y'] = -flipped_arc.points.loc[:,'y']
        flipped_arc.clockwise = not flipped_arc.clockwise
        return flipped_arc
    
    def copy(self):
        copied_arc = arc(start_point = self.points.loc['start'],
                           end_point = self.points.loc['end'],
                           center_point = self.points.loc['center'],
                           clockwise_in = self.clockwise)
        return copied_arc
    
class profile:
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
        scaled_profile = copy.deepcopy(self)
        
        for key,this_arc in scaled_profile.arcs.items():
            scaled_profile.arcs[key] = this_arc.scale(scale_factor)

        #scaled_profile.refresh()
        return scaled_profile
    
    def rotate(self,rotation_angle_rad):
        rotated_profile = copy.deepcopy(self)
        
        for key,this_arc in rotated_profile.arcs.items():
            rotated_profile.arcs[key] = this_arc.rotate(rotation_angle_rad)
            
        return rotated_profile
    
    def copy(self):
        #not finished, deepcopy doesn't work here
        copied_hold = copy.deepcopy(self)
        
    def reverse(self):
        reversed_hold = copy.deepcopy(self)

        for i,arc in enumerate(self.arcs):
            reversed_hold.arcs.iloc[i] = arc.reverse()
            
        reversed_hold.arcs = reversed_hold.arcs[::-1]
        return reversed_hold
    
    def flip(self):
        flipped_profile = copy.deepcopy(self)
        
        for key,arc in flipped_profile.arcs.items():
            flipped_profile.arcs[key] = arc.flip_vertical().reverse()
        flipped_profile.arcs = flipped_profile.arcs[::-1]
        flipped_profile.arcs.index = self.arcs.index.copy()
        return flipped_profile
    
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
        
        colors = ['orange','blue','red','yellow','pink','black']
        color_index = 0
        for key,this_arc in self.arcs.items():
            this_arc.plot_arc(axes,colors[color_index])
            plt.scatter(this_arc.points.loc['start','x'],this_arc.points.loc['start','y'],color = colors[color_index])
            color_index += 1
            
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
    
    def generate_smaller_profile(self,face_shift):
        # figure out the new top edge position
        ledge_end_vector = self.arcs['top_ledge'].points.loc[['center','end']].diff().loc['end']
        ledge_start_vector = self.arcs['top_ledge'].points.loc[['center','start']].diff().loc['start']
    
        ledge_sweep_angle = np.arcsin(cross2d(ledge_end_vector.values,ledge_start_vector.values)
                                      / (np.linalg.norm(ledge_end_vector) **2)
                                      )
    
        shift_angle = ledge_sweep_angle*face_shift
        
        ledge_edge_vector = self.arcs['top_edge'].points.loc['center'] - self.arcs['top_ledge'].points.loc['center']
        new_top_edge_position = [ledge_edge_vector['x']*np.cos(shift_angle) - ledge_edge_vector['y']*np.sin(shift_angle),
                             ledge_edge_vector['x']*np.sin(shift_angle) + ledge_edge_vector['y']*np.cos(shift_angle)] + self.arcs['top_ledge'].points.loc['center']
        
        # figure out the new bottom edge position
        ledge_end_vector = self.arcs['bottom_ledge'].points.loc[['center','end']].diff().loc['end']
        ledge_start_vector = self.arcs['bottom_ledge'].points.loc[['center','start']].diff().loc['start']
    
        ledge_sweep_angle = np.arcsin(cross2d(ledge_start_vector.values,ledge_end_vector.values)
                                      / (np.linalg.norm(ledge_end_vector) **2)
                                      )
    
        shift_angle = ledge_sweep_angle*face_shift
        ledge_edge_vector = self.arcs['bottom_edge'].points.loc['center'] - self.arcs['bottom_ledge'].points.loc['center']
        new_bottom_edge_position = [ledge_edge_vector['x']*np.cos(shift_angle) - ledge_edge_vector['y']*np.sin(shift_angle),
                             ledge_edge_vector['x']*np.sin(shift_angle) + ledge_edge_vector['y']*np.cos(shift_angle)] + self.arcs['bottom_ledge'].points.loc['center']
        
        
        smaller_profile = profile(top_edge_position = new_top_edge_position,
                               top_edge_radius = self.arcs['top_edge'].radius,
                               top_ledge_angle = self.arcs['top_ledge'].get_tangent_angle('start'),
                               top_ledge_start_height = self.arcs['top_ledge'].points.loc['start','y'],
                               bottom_edge_position = new_bottom_edge_position,
                               bottom_edge_radius = self.arcs['bottom_edge'].radius,
                               bottom_ledge_angle = self.arcs['bottom_ledge'].get_tangent_angle('end') + np.pi,
                               bottom_ledge_start_height = self.arcs['bottom_ledge'].points.loc['end','y'],
                               face_angle = self.arcs['top_face'].get_tangent_angle('end') + np.pi,
                               face_thickness = self.arcs['top_face'].points.loc['end','x'] * (1- face_shift)
                               )
        return smaller_profile

if __name__ == "__main__":
    new_profile = profile(top_edge_position = [27,33],
                                top_edge_radius = 4,
                                top_ledge_angle = 0.6,
                                top_ledge_start_height = 27,
                                bottom_edge_position = [31,-33],
                                bottom_edge_radius = 5,
                                bottom_ledge_angle = -np.pi/10,
                                bottom_ledge_start_height = -30,
                                face_angle = np.pi/2-0.2,
                                face_thickness = 20)