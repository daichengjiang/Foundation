#!/usr/bin/env python3
# Copyright (c) 2025 Xu Yang
# HKUST UAV Group
#
# Author: Xu Yang
# Affiliation: HKUST UAV Group
# Date: April 2025
# License: MIT License

import sys
import argparse
from datetime import datetime
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RadioButtons, CheckButtons
from matplotlib.gridspec import GridSpec


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize flight control logs')
    parser.add_argument('-r', '--log-file', required=True,
                        help="Path to the log file to visualize, or '-' to read from stdin")
    parser.add_argument('-d', '--display-mode', default='combined',
                        choices=['combined', 'comparison', 'separated'],
                        help="Display mode: combined (all in one), comparison (target vs actual), or separated (tabs)")
    return parser.parse_args()


def load_log_data(log_file_path):
    """Load and process log data from CSV file or stdin."""
    try:
        if log_file_path == '-':
            df = pd.read_csv(sys.stdin)
        else:
            df = pd.read_csv(log_file_path)
        df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
        start_ts = df['timestamp'].iloc[0]
        df['relative_time'] = (df['timestamp'] - start_ts) / 1000.0
        return df
    except Exception as e:
        print(f"Error loading log data: {e}", file=sys.stderr)
        sys.exit(1)


def setup_plot_style():
    """Configure global plot style."""
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = [14, 10]
    plt.rcParams['font.size'] = 10


class FlightDataVisualizer:
    def __init__(self, df, display_mode='combined'):
        self.df = df
        self.display_mode = display_mode
        self.t = df['relative_time'].to_numpy()
        self.max_time = float(self.t[-1])
        
        # Data categories and their fields - separate thrust
        self.categories = {
            'Control Angles': ['target_roll', 'target_pitch', 'target_yaw_rate'],
            'Thrust': ['thrust'],
            'Position': ['pos_x', 'pos_y', 'pos_z'],
            'Velocity': ['vel_x', 'vel_y', 'vel_z'],
            'Attitude': ['roll', 'pitch', 'yaw'],
            'Angular Velocity': ['gyro_x', 'gyro_y', 'gyro_z'],
            'Acceleration': ['acc_x', 'acc_y', 'acc_z']
        }
        
        # Pairs for comparison plots
        self.comparison_pairs = [
            ('target_roll', 'roll'),
            ('target_pitch', 'pitch'),
            ('target_yaw_rate', 'gyro_z')  # assuming gyro_z is the closest to actual yaw rate
        ]
        
        # Display names for each field
        self.display_names = {
            'pos_x': 'X Position (m)', 'pos_y': 'Y Position (m)', 'pos_z': 'Z Position (m)',
            'vel_x': 'X Velocity (m/s)', 'vel_y': 'Y Velocity (m/s)', 'vel_z': 'Z Velocity (m/s)',
            'roll': 'Roll (°)', 'pitch': 'Pitch (°)', 'yaw': 'Yaw (°)',
            'target_roll': 'Target Roll (°)', 'target_pitch': 'Target Pitch (°)', 
            'target_yaw_rate': 'Target Yaw Rate (°/s)', 'thrust': 'Thrust',
            'gyro_x': 'Roll Rate (°/s)', 'gyro_y': 'Pitch Rate (°/s)', 'gyro_z': 'Yaw Rate (°/s)',
            'acc_x': 'X Accel (m/s²)', 'acc_y': 'Y Accel (m/s²)', 'acc_z': 'Z Accel (m/s²)'
        }
        
        # Colors for each axis and comparison
        self.axis_colors = {
            'x': 'tab:red',
            'y': 'tab:green',
            'z': 'tab:blue',
            'roll': 'tab:orange',
            'pitch': 'tab:purple',
            'yaw': 'tab:brown',
            'target': 'tab:red',
            'actual': 'tab:blue',
            'thrust': 'tab:olive'
        }
        
        # Initial view settings
        self.init_start = 0.0
        self.init_window = min(30.0, self.max_time)
        
        # Create the figure based on display mode
        if display_mode == 'combined':
            self.create_combined_view()
        elif display_mode == 'comparison':
            self.create_comparison_view()
        else:
            self.create_tabbed_view()
            
    def create_combined_view(self):
        """Create a view with all plots in one window"""
        # Create figure with GridSpec for better layout control
        self.fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(7, 2, height_ratios=[1, 1, 1, 1, 1, 1, 0.5], figure=self.fig)
        
        # Create axes for each data category
        self.axes = {}
        self.axes['Position'] = self.fig.add_subplot(gs[0, 0])
        self.axes['Velocity'] = self.fig.add_subplot(gs[1, 0], sharex=self.axes['Position'])
        self.axes['Attitude'] = self.fig.add_subplot(gs[2, 0], sharex=self.axes['Position'])
        self.axes['Angular Velocity'] = self.fig.add_subplot(gs[3, 0], sharex=self.axes['Position'])
        self.axes['Acceleration'] = self.fig.add_subplot(gs[4, 0], sharex=self.axes['Position'])
        
        # Separate Control Angles and Thrust
        self.axes['Control Angles'] = self.fig.add_subplot(gs[0, 1], sharex=self.axes['Position'])
        self.axes['Thrust'] = self.fig.add_subplot(gs[1, 1], sharex=self.axes['Position'])
        
        # XY plot for position
        self.axes['XY Plot'] = self.fig.add_subplot(gs[2:5, 1])
        
        # Plot data for each category
        for category, ax in self.axes.items():
            if category == 'XY Plot':
                self.plot_xy_data(ax)
            else:
                self.plot_category(category, ax)
                
        # Add shared x-label
        self.fig.text(0.5, 0.02, 'Time (seconds)', ha='center', va='center')
        
        # Store initial y-limits
        self.initial_ylims = {cat: ax.get_ylim() for cat, ax in self.axes.items() if cat != 'XY Plot'}
        
        # Add sliders and controls
        self.add_controls()
        
        # Set title and adjust layout
        start_str = self.df['time'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')
        end_str = self.df['time'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')
        duration = self.max_time
        self.fig.suptitle(f"Flight Log: {start_str} to {end_str} (Duration: {duration:.2f}s)", fontsize=12)
        
        plt.tight_layout(rect=[0, 0.2, 1, 0.97])
    
    def create_comparison_view(self):
        """Create a view focusing on control tracking performance"""
        # Create figure with GridSpec for better layout control
        self.fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(5, 2, height_ratios=[1, 1, 1, 1, 0.5], figure=self.fig)
        
        # Create axes for each control comparison
        self.axes = {}
        
        # Dedicated large plots for comparing target vs actual
        self.axes['Roll Comparison'] = self.fig.add_subplot(gs[0, :])
        self.axes['Pitch Comparison'] = self.fig.add_subplot(gs[1, :], sharex=self.axes['Roll Comparison'])
        self.axes['Yaw Rate Comparison'] = self.fig.add_subplot(gs[2, :], sharex=self.axes['Roll Comparison'])
        
        # Thrust has its own dedicated plot
        self.axes['Thrust'] = self.fig.add_subplot(gs[3, 0], sharex=self.axes['Roll Comparison'])
        
        # XY plot for position
        self.axes['XY Plot'] = self.fig.add_subplot(gs[3, 1])
        
        # Plot each comparison
        self.plot_comparison('Roll Comparison', 'target_roll', 'roll', self.axes['Roll Comparison'])
        self.plot_comparison('Pitch Comparison', 'target_pitch', 'pitch', self.axes['Pitch Comparison'])
        self.plot_comparison('Yaw Rate Comparison', 'target_yaw_rate', 'gyro_z', self.axes['Yaw Rate Comparison'])
        
        # Plot thrust and XY
        self.plot_category('Thrust', self.axes['Thrust'])
        self.plot_xy_data(self.axes['XY Plot'])
        
        # Add shared x-label
        self.fig.text(0.5, 0.02, 'Time (seconds)', ha='center', va='center')
        
        # Store initial y-limits
        self.initial_ylims = {cat: ax.get_ylim() for cat, ax in self.axes.items() if cat != 'XY Plot'}
        
        # Add sliders and controls
        self.add_controls()
        
        # Set title and adjust layout
        start_str = self.df['time'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')
        end_str = self.df['time'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')
        duration = self.max_time
        self.fig.suptitle(f"Control Tracking Performance: {start_str} to {end_str} (Duration: {duration:.2f}s)", fontsize=12)
        
        plt.tight_layout(rect=[0, 0.2, 1, 0.97])
        
    def create_tabbed_view(self):
        """Create a tabbed view with radio buttons to switch between data categories"""
        self.fig = plt.figure(figsize=(14, 10))
        
        # Create a better layout with fixed space for controls at bottom
        # Leave more space between elements for better appearance
        plot_height = 0.65  # Main plot takes 65% of the figure
        radio_height = 0.12  # Radio buttons take 12%
        
        # Main plot area - increase bottom margin to 0.3 to leave room for x-axis labels
        plot_rect = [0.1, 0.3, 0.8, plot_height]  # [left, bottom, width, height]
        self.main_ax = self.fig.add_axes(plot_rect)
        
        # Radio buttons area - move down to 0.15 to create space between plot and buttons
        radio_rect = [0.1, 0.15, 0.8, radio_height]
        radio_ax = self.fig.add_axes(radio_rect)
        
        # Create radio buttons with all options
        tab_options = list(self.categories.keys()) + ['XY Plot']
        # Add comparison plots to the radio options
        tab_options += ['Roll Comparison', 'Pitch Comparison', 'Yaw Rate Comparison']
        self.radio = RadioButtons(radio_ax, tab_options)
        self.radio.on_clicked(self.update_tab)
        
        # Add controls - no need to pass GridSpec, we'll position manually
        self.add_controls_tabbed()
        
        # Initial y-limits will be set on first tab display
        self.initial_ylims = {}
        
        # Set title and show first tab
        start_str = self.df['time'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')
        end_str = self.df['time'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')
        duration = self.max_time
        self.fig.suptitle(f"Flight Log: {start_str} to {end_str} (Duration: {duration:.2f}s)", fontsize=12)
        
        # Add an x-axis label for time directly to the main plot
        self.main_ax.set_xlabel('Time (seconds)', labelpad=10)
        
        # Show first tab
        self.update_tab(tab_options[0])
        
    def plot_category(self, category, ax):
        """Plot data for a specific category"""
        if category not in self.categories:
            return
            
        fields = self.categories[category]
        for field in fields:
            if field in self.df.columns:
                color = self.get_field_color(field)
                ax.plot(self.t, self.df[field].to_numpy(), '-', 
                        label=self.display_names.get(field, field),
                        linewidth=1.5, color=color)
        
        ax.set_ylabel(category)
        ax.legend(loc='upper right')
        ax.grid(True)
        ax.set_xlim(self.init_start, self.init_start + self.init_window)
    
    def plot_comparison(self, title, target_field, actual_field, ax):
        """Plot comparison between target and actual values"""
        if target_field in self.df.columns and actual_field in self.df.columns:
            # Target line
            ax.plot(self.t, self.df[target_field].to_numpy(), '-', 
                    label=f"Target ({self.display_names.get(target_field, target_field)})",
                    linewidth=2.0, color=self.axis_colors['target'])
            
            # Actual line
            ax.plot(self.t, self.df[actual_field].to_numpy(), '--', 
                    label=f"Actual ({self.display_names.get(actual_field, actual_field)})",
                    linewidth=1.5, color=self.axis_colors['actual'])
            
            # Calculate error metrics
            error = self.df[target_field] - self.df[actual_field]
            mean_error = error.mean()
            rmse = np.sqrt((error**2).mean())
            
            # Display error metrics in title
            ax.set_title(f"{title}: Mean Error={mean_error:.2f}°, RMSE={rmse:.2f}°")
            
            # Add shaded error area
            ax.fill_between(self.t, 
                           self.df[target_field].to_numpy(), 
                           self.df[actual_field].to_numpy(), 
                           color='gray', alpha=0.3, label='Error')
        
        ax.legend(loc='upper right')
        ax.grid(True)
        ax.set_xlim(self.init_start, self.init_start + self.init_window)
        
    def plot_xy_data(self, ax):
        """Plot XY position data"""
        if 'pos_x' in self.df.columns and 'pos_y' in self.df.columns:
            x = self.df['pos_x'].to_numpy()
            y = self.df['pos_y'].to_numpy()
            
            # Get colormap based on time
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Color by time for trajectory visualization
            norm = plt.Normalize(self.t.min(), self.t.max())
            lc = plt.matplotlib.collections.LineCollection(segments, cmap='viridis', norm=norm)
            lc.set_array(self.t)
            
            line = ax.add_collection(lc)
            self.fig.colorbar(line, ax=ax, label='Time (s)')
            
            # Plot start and end points
            ax.plot(x[0], y[0], 'go', markersize=8, label='Start')
            ax.plot(x[-1], y[-1], 'ro', markersize=8, label='End')
            
            # Set equal aspect to prevent distortion
            ax.set_aspect('equal', 'datalim')
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            ax.grid(True)
            ax.legend(loc='upper right')
            
    def get_field_color(self, field):
        """Get color for a specific field based on axis"""
        for axis, color in self.axis_colors.items():
            if axis in field:
                return color
        return 'tab:blue'  # Default color
    
    def add_controls(self, gs_slot=None):
        """Add sliders and buttons for interaction for combined/comparison views"""
        # For tabbed view, we now use the dedicated add_controls_tabbed method
        if gs_slot is not None:
            return
            
        axcolor = 'lightgoldenrodyellow'
        
        # For combined and comparison views, create axes at bottom
        ax_y = self.fig.add_axes([0.15, 0.08, 0.65, 0.02], facecolor=axcolor)
        ax_start = self.fig.add_axes([0.15, 0.05, 0.65, 0.02], facecolor=axcolor)
        ax_window = self.fig.add_axes([0.15, 0.02, 0.50, 0.02], facecolor=axcolor)
        ax_reset = self.fig.add_axes([0.70, 0.02, 0.10, 0.02])
        
        self.slider_y = Slider(ax_y, 'Y Scale', 0.1, 3.0, valinit=1.0, valstep=0.1)
        self.slider_start = Slider(ax_start, 'Start', 0.0, max(0.1, self.max_time - 0.1), 
                                  valinit=self.init_start, valstep=0.1)
        self.slider_window = Slider(ax_window, 'Window', 0.1, self.max_time, 
                                   valinit=self.init_window, valstep=0.1)
        
        btn = Button(ax_reset, 'Reset View')
        
        # Connect events
        self.slider_y.on_changed(self.update_view)
        self.slider_start.on_changed(self.update_view)
        self.slider_window.on_changed(self.update_view)
        btn.on_clicked(self.reset_view)
        
        # Navigation hint
        self.fig.text(0.01, 0.01,
                     "Navigation: Drag to pan, Scroll to zoom, Right-click drag to zoom region",
                     fontsize=8, color='gray')
        
        # Add keyboard shortcut hints
        self.fig.text(0.01, 0.03,
                    "Shortcuts: [m] Combined view, [c] Comparison view, [t] Tabbed view",
                    fontsize=8, color='gray')
    
    def add_controls_tabbed(self):
        """Add sliders and buttons specifically for tabbed view"""
        axcolor = 'lightgoldenrodyellow'
        
        # Slider positions - moved up from bottom edge by increasing y values
        # Bottom slider now at y=0.02 instead of 0.0
        y_slider_rect = [0.25, 0.09, 0.65, 0.03]
        start_slider_rect = [0.25, 0.06, 0.65, 0.03]
        window_slider_rect = [0.25, 0.03, 0.40, 0.03]
        reset_btn_rect = [0.70, 0.03, 0.15, 0.03]
        
        # Create the slider and button axes
        ax_y = self.fig.add_axes(y_slider_rect, facecolor=axcolor)
        ax_start = self.fig.add_axes(start_slider_rect, facecolor=axcolor)
        ax_window = self.fig.add_axes(window_slider_rect, facecolor=axcolor)
        ax_reset = self.fig.add_axes(reset_btn_rect)
        
        # Create sliders and buttons
        self.slider_y = Slider(ax_y, 'Y Scale', 0.1, 3.0, valinit=1.0, valstep=0.1)
        self.slider_start = Slider(ax_start, 'Start', 0.0, max(0.1, self.max_time - 0.1), 
                                  valinit=self.init_start, valstep=0.1)
        self.slider_window = Slider(ax_window, 'Window', 0.1, self.max_time, 
                                   valinit=self.init_window, valstep=0.1)
        
        btn = Button(ax_reset, 'Reset View')
        
        # Connect events
        self.slider_y.on_changed(self.update_view_tabbed)
        self.slider_start.on_changed(self.update_view_tabbed)
        self.slider_window.on_changed(self.update_view_tabbed)
        btn.on_clicked(self.reset_view_tabbed)  # Use dedicated reset method for tabbed view
        
        # Navigation hint - moved up slightly
        self.fig.text(0.01, 0.02,
                     "Navigation: Drag to pan, Scroll to zoom, Right-click drag to zoom region",
                     fontsize=8, color='gray')
        
        # Add keyboard shortcut hints
        self.fig.text(0.01, 0.04,
                    "Shortcuts: [m] Combined view, [c] Comparison view, [t] Tabbed view",
                    fontsize=8, color='gray')

    def reset_view_tabbed(self, event):
        """Reset sliders to initial values for tabbed view"""
        try:
            # Store the current radio selection
            current_tab = self.radio.value_selected
            
            # Reset all sliders
            self.slider_y.reset()
            self.slider_start.reset()
            self.slider_window.reset()
            
            # Force redraw the current tab with reset values
            self.update_tab(current_tab)
            
            # For debugging
            print(f"Tabbed view reset. Y-scale: {self.slider_y.val}, Start: {self.slider_start.val}, Window: {self.slider_window.val}")
        except Exception as e:
            print(f"Error in reset_view_tabbed: {e}")

    def reset_view(self, event):
        """Reset sliders to initial values"""
        try:
            # Reset all sliders
            self.slider_y.reset()
            self.slider_start.reset()
            self.slider_window.reset()
            
            # Redraw after reset - use the appropriate update method
            if self.display_mode == 'separated':
                self.update_view_tabbed()
            else:
                self.update_view()
                
            # For debugging
            print(f"View reset. Mode: {self.display_mode}, Y-scale: {self.slider_y.val}, Start: {self.slider_start.val}, Window: {self.slider_window.val}")
        except Exception as e:
            print(f"Error in reset_view: {e}")

    def update_tab(self, label):
        """Update the displayed tab in tabbed view"""
        # Clear the main axis for new content
        self.main_ax.clear()
        
        try:
            # Special handling for comparison plots
            if label == 'Roll Comparison':
                self.plot_comparison('Roll Comparison', 'target_roll', 'roll', self.main_ax)
            elif label == 'Pitch Comparison':
                self.plot_comparison('Pitch Comparison', 'target_pitch', 'pitch', self.main_ax)
            elif label == 'Yaw Rate Comparison':
                self.plot_comparison('Yaw Rate Comparison', 'target_yaw_rate', 'gyro_z', self.main_ax)
            elif label == 'XY Plot':
                self.plot_xy_data(self.main_ax)
            else:
                self.plot_category(label, self.main_ax)
                
            # Store initial y-limits for this category
            self.initial_ylims[label] = self.main_ax.get_ylim()
            
            # Always add x-label for time
            if label != 'XY Plot':
                self.main_ax.set_xlabel('Time (seconds)', labelpad=10)
                
            # Update the view to apply current slider settings
            self.update_view_tabbed()
            
        except Exception as e:
            print(f"Error updating tab to {label}: {e}")
    
    def update_view_tabbed(self, val=None):
        """Special update function for tabbed view to prevent freezing"""
        # Disable slider events to prevent recursive calls
        self.slider_y.eventson = False
        self.slider_start.eventson = False
        self.slider_window.eventson = False
        
        try:
            # Get current values
            start = self.slider_start.val
            window = self.slider_window.val
            y_scale = self.slider_y.val
            end = min(start + window, self.max_time)
            
            # Get current tab
            current_category = self.radio.value_selected
            
            # Update plot limits
            if current_category == 'XY Plot':
                # No need to update x-limits for XY plot
                pass
            else:
                # Set x-axis limits
                self.main_ax.set_xlim(start, end)
                
                # Update y-axis limits if we have initial limits
                if current_category in self.initial_ylims:
                    ylim = self.initial_ylims[current_category]
                    mid = (ylim[0] + ylim[1]) / 2.0
                    half_range = (ylim[1] - ylim[0]) / 2.0 * y_scale
                    self.main_ax.set_ylim(mid - half_range, mid + half_range)
            
            # Redraw canvas
            self.fig.canvas.draw_idle()
        
        except Exception as e:
            print(f"Error updating view: {e}")
        
        finally:
            # Always re-enable slider events
            self.slider_y.eventson = True
            self.slider_start.eventson = True
            self.slider_window.eventson = True
    
    def update_view(self, val=None):
        """Update the view based on slider values (for combined/comparison modes)"""
        # Prevent recursive draws
        self.slider_y.eventson = self.slider_start.eventson = self.slider_window.eventson = False
        
        try:
            start = self.slider_start.val
            window = self.slider_window.val
            y_scale = self.slider_y.val
            end = min(start + window, self.max_time)
            
            # Only handle combined and comparison modes here
            if self.display_mode == 'combined' or self.display_mode == 'comparison':
                axes_to_update = self.axes
                
                for category, ax in axes_to_update.items():
                    if category != 'XY Plot':
                        ax.set_xlim(start, end)
                        if category in self.initial_ylims:
                            mid = (self.initial_ylims[category][0] + self.initial_ylims[category][1]) / 2.0
                            half_range = (self.initial_ylims[category][1] - self.initial_ylims[category][0]) / 2.0 * y_scale
                            ax.set_ylim(mid - half_range, mid + half_range)
            
            self.fig.canvas.draw_idle()
        
        except Exception as e:
            print(f"Error updating view: {e}")
            
        finally:
            self.slider_y.eventson = self.slider_start.eventson = self.slider_window.eventson = True

def main():
    args = parse_args()
    df = load_log_data(args.log_file)
    setup_plot_style()
    
    # Set up matplotlib for better interactivity
    plt.ion()
    
    # Create visualizer instance
    visualizer = FlightDataVisualizer(df, args.display_mode)
    
    # Connect key events with proper error handling
    def on_key(event):
        try:
            if event.key == 'r':
                visualizer.fig.canvas.draw()
            elif event.key == 'c':
                # Switch to comparison view on 'c' key
                plt.close()
                visualizer = FlightDataVisualizer(df, 'comparison')
                visualizer.fig.canvas.mpl_connect('key_press_event', on_key)
            elif event.key == 'm':
                # Switch to combined view on 'm' key
                plt.close()
                visualizer = FlightDataVisualizer(df, 'combined')
                visualizer.fig.canvas.mpl_connect('key_press_event', on_key)
            elif event.key == 't':
                # Switch to tabbed view on 't' key
                plt.close()
                visualizer = FlightDataVisualizer(df, 'separated')
                visualizer.fig.canvas.mpl_connect('key_press_event', on_key)
        except Exception as e:
            print(f"Error handling key event: {e}")
            
    visualizer.fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.show(block=True)

if __name__ == "__main__":
    main()
