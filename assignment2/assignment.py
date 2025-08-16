import os
import sys
import re
import shelve
import json
import heapq
import random
import getpass
import logging
from copy import deepcopy
from logging.handlers import RotatingFileHandler
from rich.console import Console
from rich.table import Table
import pyfiglet
from tabulate import tabulate
import pandas as pd
from fpdf import FPDF
from datetime import datetime, timedelta
import bcrypt
from typing import List, Optional
import google.generativeai as genai
import plotly.express as px
import cv2
import pickle
import shutil
import urllib.request
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from collections import Counter
import threading
import time
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from cryptography.fernet import Fernet
import tkinter as tk
from tkinter import ttk
from flask import Flask, render_template, jsonify
import webbrowser
from collections import defaultdict
from pathlib import Path


load_dotenv()


# Initialize rich console with wider display
console = Console(width=140)



# Configure logging
def setup_logging():
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # System logger
    system_logger = logging.getLogger('system')
    system_logger.setLevel(logging.INFO)

    # Action logger
    action_logger = logging.getLogger('actions')
    action_logger.setLevel(logging.INFO)

    # Rotating handler for system logs
    system_handler = RotatingFileHandler(
        'logs/system.log', maxBytes=1024 * 1024, backupCount=5
    )

    # File handler for action logs
    action_handler = logging.FileHandler('logs/employee.log.txt')

    # Formatters
    system_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    action_formatter = logging.Formatter(
        '%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    system_handler.setFormatter(system_formatter)
    action_handler.setFormatter(action_formatter)

    system_logger.addHandler(system_handler)
    action_logger.addHandler(action_handler)

    return system_logger, action_logger


system_logger, action_logger = setup_logging()


def __init__(self, system, face_db_file='face_data.dat'):
    self.known_faces = {}
    self.face_db_file = face_db_file
    self.tolerance = 0.5  # Lower is more strict (changed from 0.6)
    self.min_confidence = 0.75  # Increased from 0.7
    self.face_detector = None
    self.face_recognizer = None
    self.model_file = 'openface_nn4.small2.v1.t7'
    self.system = system
    self._initialize_models()
    self._load_face_data()


class Employee:
    def __init__(self, name, employee_id, email, department, full_time_status, enrolled_programmes=None, encrypted_email=True):
        self.name = name
        self.employee_id = employee_id
        self.email = self.system.encrypt_sensitive_data(email.lower()) if hasattr(self, 'system') else email.lower()
        self.department = department
        self.full_time_status = full_time_status
        self.enrolled_programmes = enrolled_programmes if enrolled_programmes else []
        self.enrollment_history = ProgrammeHistory()
        self.current_programmes = []
        self.password = None
        self.feedback = []
        self.pending_requests = []
        self.face_registered = False

    def verify_password(self, password):
        if not self.password:
            return False
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

    def to_dict(self):
        return {
            'name': self.name,
            'employee_id': self.employee_id,
            'email': self.email,
            'department': self.department,
            'full_time_status': self.full_time_status,
            'enrolled_programmes': self.enrolled_programmes,
            'password': self.password,
            'feedback': self.feedback,
            'pending_requests': self.pending_requests,
            'current_programmes': self.current_programmes,
            'face_registered': self.face_registered
        }

    def verify_data_integrity(self):
        """Verify that saved data can be loaded correctly"""
        try:
            db_path = os.path.join('data', 'employee_data')
            with shelve.open(db_path, flag='r') as db:
                if 'employees' not in db:
                    return False
                test_employee = next(iter(db['employees'].values()), None)
                # Check if required fields exist
                required_fields = ['name', 'employee_id', 'email', 'department']
                return all(field in test_employee for field in required_fields)
        except:
            return False

    def add_training_programme(self, programme):
        """Enroll in a new programme with proper validation"""
        if not programme:
            return False

        programme_lower = programme.lower()

        # Check if already enrolled in this program (case insensitive)
        if any(p.lower() == programme_lower for p in self.current_programmes):
            console.print(f"[yellow]Already enrolled in: {programme}[/yellow]")
            return False

        # Add to current programs list
        self.current_programmes.append(programme)

        # Add to enrolled_programmes list for backward compatibility
        if programme not in self.enrolled_programmes:
            self.enrolled_programmes.append(programme)

        # Add to enrollment history
        self.enrollment_history.add_enrollment(programme)

        action_logger.info(f"Employee {self.employee_id} enrolled in {programme}")
        return True

    def complete_programme(self, programme):
        """Mark a programme as completed"""
        enrollment = self.enrollment_history.find_enrollment(programme)
        if enrollment:
            enrollment.mark_completed()
            action_logger.info(f"Employee {self.employee_id} completed {programme}")
            return True
        return False

    def remove_training_programme(self, programme):
        """Remove a programme from enrollment and update history"""
        programme_lower = programme.lower()
        removed = False

        # Remove from current programmes lists
        for lst in [self.enrolled_programmes, getattr(self, 'current_programmes', [])]:
            for i, prog in enumerate(lst[:]):
                if prog.lower() == programme_lower:
                    lst.pop(i)
                    removed = True

        # Update enrollment history to mark as dropped
        if removed:
            enrollment = self.enrollment_history.find_enrollment(programme)
            if enrollment:
                enrollment.mark_dropped()
            else:
                # Add to history if not found (shouldn't happen but just in case)
                self.enrollment_history.add_enrollment(programme, status="Dropped")
            action_logger.info(f"Employee {self.employee_id} unenrolled from {programme}")
            return True
        return False

    def display_details(self):
        status = "Full-time" if self.full_time_status else "Part-time"
        programmes = "\n".join(f"• {prog}" for prog in self.enrolled_programmes) if self.enrolled_programmes else "None"

        table = Table(title=f"Employee Details - {self.name}", show_header=True, header_style="bold blue", expand=True)
        table.add_column("Field", style="dim", width=20)
        table.add_column("Value", width=60)

        table.add_row("Employee ID", str(self.employee_id))
        table.add_row("Name", self.name)
        table.add_row("Email", self.email)
        table.add_row("Department", self.department)
        table.add_row("Status", status)
        table.add_row("Current Programmes", programmes)

        # Add enrollment history section
        history_table = Table(title="Enrollment History", show_header=True, header_style="bold magenta")
        history_table.add_column("Programme")
        history_table.add_column("Status")
        history_table.add_column("Date")

        current = self.enrollment_history.head
        while current:
            status_color = "green" if current.status == "Completed" else "red" if current.status == "Dropped" else "yellow"
            history_table.add_row(
                current.programme_name,
                f"[{status_color}]{current.status}[/{status_color}]",
                current.enrollment_date.strftime("%Y-%m-%d")
            )
            current = current.next

        console.print(table)
        if self.enrollment_history.head:  # Only show history if exists
            console.print(history_table)

    def add_feedback(self, programme, feedback_text, rating):
        """Add feedback to a programme"""
        enrollment = self.enrollment_history.find_enrollment(programme)
        if enrollment:
            enrollment.add_feedback(feedback_text, rating)
            action_logger.info(f"Employee {self.employee_id} submitted feedback for {programme}")
            return True
        return False

    def request_enrollment(self, system, programme=None):
        """Request enrollment in a program with AI recommendations"""
        if programme is None:
            # Get AI recommendations if no specific program is requested
            recommendations = system.program_advisor.get_recommendations(self)

            if recommendations:
                console.print("\n[bold]Recommended Programs:[/bold]")
                for i, rec in enumerate(recommendations, 1):
                    console.print(f"{i}. {rec}")

                # Allow selecting a recommendation
                choice = system.get_input("\nSelect recommendation to request (number) or enter program name: ")
                if choice:
                    try:
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < len(recommendations):
                            programme = recommendations[choice_idx]
                    except ValueError:
                        # If not a number, treat as direct program name
                        programme = choice
            else:
                programme = system.get_input("Enter program name to request: ", required=True)
                if not programme:
                    return False

        programme_lower = programme.lower()
        if (not any(p.lower() == programme_lower for p in self.enrolled_programmes) and
                not any(req['programme'].lower() == programme_lower for req in self.pending_requests)):
            self.pending_requests.append({
                'programme': programme,
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'status': 'Pending'
            })
            action_logger.info(f"Employee {self.employee_id} requested enrollment in {programme}")
            return True
        return False

    def to_dict(self):
        return {
            'name': self.name,
            'employee_id': self.employee_id,
            'email': self.email,
            'department': self.department,
            'full_time_status': self.full_time_status,
            'enrolled_programmes': self.enrolled_programmes,
            'password': self.password,
            'feedback': self.feedback,
            'pending_requests': self.pending_requests,
            'current_programmes': self.current_programmes
        }


class EmployeeRequest:
    def __init__(self, employee_id, request_type, priority_level, request_details):
        self.employee_id = employee_id
        self.request_type = request_type
        self.priority_level = priority_level
        self.request_details = request_details
        self.timestamp = datetime.now()

    def __lt__(self, other):
        """Comparison for priority queue - lower priority_level comes first, then older timestamp"""
        if self.priority_level == other.priority_level:
            return self.timestamp < other.timestamp
        return self.priority_level < other.priority_level

    def __repr__(self):
        return (f"EmployeeRequest(employee_id={self.employee_id}, type={self.request_type}, "
                f"priority={self.priority_level}, details={self.request_details}, "
                f"timestamp={self.timestamp.strftime('%Y-%m-%d %H:%M:%S')})")

    def to_dict(self):
        """Convert request to dictionary for serialization"""
        return {
            'employee_id': self.employee_id,
            'request_type': self.request_type,
            'priority_level': self.priority_level,
            'request_details': self.request_details,
            'timestamp': self.timestamp
        }


class EmployeeRequestQueue:
    def __init__(self):
        self.requests = []
        self.processed_requests = []

    def add_request(self, request):
        heapq.heappush(self.requests, request)

    def get_next_request(self):
        if self.requests:
            return heapq.heappop(self.requests)
        return None

    def peek_next_request(self):
        if self.requests:
            return self.requests[0]
        return None

    def is_empty(self):
        return len(self.requests) == 0

    def size(self):
        return len(self.requests)

    def get_requests_by_type(self, request_type):
        return [req for req in self.requests if req.request_type == request_type]

    def get_requests_by_priority(self, priority_level):
        return [req for req in self.requests if req.priority_level == priority_level]


class RequestHistory:
    def __init__(self):
        self.undo_stack = []
        self.redo_stack = []
        self.max_stack_size = 100
        self.action_types = {
            'add': self._undo_add,
            'process': self._undo_process,
            'modify': self._undo_modify,
            'priority_change': self._undo_priority_change
        }

    def record_action(self, action_type: str, request: EmployeeRequest, prev_state=None):
        """Enhanced action recording with state capture"""
        if len(self.undo_stack) >= self.max_stack_size:
            self.undo_stack.pop(0)

        action_record = {
            'type': action_type,
            'request': deepcopy(request),
            'timestamp': datetime.now(),
            'prev_state': prev_state
        }
        self.undo_stack.append(action_record)
        self.redo_stack = []

    def _undo_add(self, request_queue, action_record):
        """Undo an add action by removing the request"""
        if action_record['request'] in request_queue.requests:
            request_queue.requests.remove(action_record['request'])
            heapq.heapify(request_queue.requests)

    def _undo_process(self, request_queue, action_record):
        """Undo a process action by re-adding the request"""
        request_queue.add_request(action_record['request'])

    def _undo_modify(self, request_queue, action_record):
        """Undo a modify action by restoring previous state"""
        if action_record['prev_state']:
            # Find the request in the queue and restore its previous state
            for i, req in enumerate(request_queue.requests):
                if req.employee_id == action_record['request'].employee_id and \
                        req.request_type == action_record['request'].request_type:
                    request_queue.requests[i] = action_record['prev_state']
                    heapq.heapify(request_queue.requests)
                    break

    def _undo_priority_change(self, request_queue, action_record):
        """Undo a priority change by restoring previous priority"""
        if action_record['prev_state']:
            # Find the request and restore its previous priority
            for req in request_queue.requests:
                if req.employee_id == action_record['request'].employee_id and \
                        req.request_type == action_record['request'].request_type:
                    req.priority_level = action_record['prev_state']['priority']
                    heapq.heapify(request_queue.requests)
                    break

    def undo(self, request_queue: EmployeeRequestQueue) -> bool:
        """Perform undo operation"""
        try:
            if not self.undo_stack:
                return False

            action_record = self.undo_stack.pop()
            handler = self.action_types.get(action_record['type'])
            if handler:
                handler(request_queue, action_record)
                self.redo_stack.append(action_record)
                return True
            return False
        except Exception as e:
            system_logger.error(f"Undo failed: {str(e)}")
            return False

    def redo(self, request_queue: EmployeeRequestQueue) -> bool:
        """Perform redo operation"""
        try:
            if not self.redo_stack:
                return False

            action_record = self.redo_stack.pop()
            # For redo, we need to reverse the undo action
            reverse_actions = {
                'add': self._undo_process,  # Redo add = process
                'process': self._undo_add,  # Redo process = add
                'modify': self._undo_modify,  # Redo modify is another modify
                'priority_change': self._undo_priority_change
            }
            handler = reverse_actions.get(action_record['type'])
            if handler:
                handler(request_queue, action_record)
                self.undo_stack.append(action_record)
                return True
            return False
        except Exception as e:
            system_logger.error(f"Redo failed: {str(e)}")
            return False


class DepartmentTreeNode:
    """Tree node representing department hierarchy"""

    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.employees = []

    def add_child(self, child_name):
        """Add a child department node"""
        child = DepartmentTreeNode(child_name, self)
        self.children.append(child)
        return child

    def add_employee(self, employee):
        """Add employee to this department node"""
        self.employees.append(employee)

    def find_department(self, dept_name):
        """Find department node by name (BFS search)"""
        from collections import deque
        queue = deque()
        queue.append(self)

        while queue:
            current = queue.popleft()
            if current.name.lower() == dept_name.lower():
                return current
            for child in current.children:
                queue.append(child)
        return None

    def print_tree(self, level=0):
        """Print department tree structure"""
        prefix = "    " * level
        console.print(f"{prefix}└─ {self.name} ({len(self.employees)} employees)")
        for child in self.children:
            child.print_tree(level + 1)


class EnhancedDepartmentTreeNode(DepartmentTreeNode):
    def __init__(self, name, parent=None):
        super().__init__(name, parent)
        self.programmes = []  # Department-specific programmes
        self.manager = None  # Department manager
        self.budget = 0  # Training budget

    def add_programme(self, programme_name, cost=0):
        """Add training programme to department"""
        if programme_name not in [p['name'] for p in self.programmes]:
            self.programmes.append({
                'name': programme_name,
                'cost': cost,
                'created_at': datetime.now()
            })
            return True
        return False

    def set_manager(self, employee_id):
        """Assign department manager"""
        self.manager = employee_id

    def get_training_budget_utilization(self):
        """Calculate budget utilization"""
        total_cost = sum(p['cost'] for p in self.programmes)
        return (total_cost / self.budget) * 100 if self.budget else 0

    def print_detailed_tree(self, level=0):
        """Enhanced tree visualization with more details"""
        prefix = "    " * level
        manager_name = "None"
        if self.manager and self.manager in self.system.employees:
            manager_name = self.system.employees[self.manager].name

        console.print(
            f"{prefix}└─ {self.name} "
            f"(Employees: {len(self.employees)}, "
            f"Programmes: {len(self.programmes)}, "
            f"Manager: {manager_name})"
        )
        for child in self.children:
            child.print_detailed_tree(level + 1)


class Programme:
    def __init__(self, name, description, duration_hours, prerequisites=None):
        self.name = name
        self.description = description
        self.duration_hours = duration_hours
        self.prerequisites = prerequisites if prerequisites else []

    def can_enroll(self, employee):
        """Check if employee meets prerequisites"""
        completed = [
            node.programme_name
            for node in employee.enrollment_history
            if node.status == "Completed"
        ]
        return all(req in completed for req in self.prerequisites)


class ProgrammeEnrollmentNode:
    """Node for linked list tracking enrollment history"""

    def __init__(self, programme_name, enrollment_date, status="Enrolled"):
        self.programme_name = programme_name
        self.enrollment_date = enrollment_date
        self.status = status  # 'Enrolled', 'Completed', 'Dropped'
        self.next = None
        self.completion_date = None  # Will be set when completed or dropped
        self.feedback = None
        self.rating = None

    def mark_dropped(self):
        """Mark this program as dropped"""
        self.status = "Dropped"
        self.completion_date = datetime.now()

    def mark_completed(self):
        """Mark this program as completed"""
        self.status = "Completed"
        self.completion_date = datetime.now()

    def add_feedback(self, feedback_text, rating):
        """Add feedback to this enrollment"""
        self.feedback = feedback_text
        self.rating = rating


class ProgrammeHistory:
    """Linked list tracking employee's programme history"""

    def __init__(self):
        self.head = None
        self.tail = None
        self.count = 0

    def add_enrollment(self, programme_name, status='Enrolled'):
        """Add new enrollment to history"""
        new_node = ProgrammeEnrollmentNode(
            programme_name,
            datetime.now(),
            status
        )

        if not self.head:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
        self.count += 1
        return new_node

    def find_enrollment(self, programme_name):
        """Find a specific enrollment by programme name"""
        current = self.head
        while current:
            if current.programme_name.lower() == programme_name.lower():
                return current
            current = current.next
        return None

    def display_history(self):
        """Display enrollment history with clear status indicators"""
        current = self.head
        if not current:
            console.print("[yellow]No enrollment history found.[/yellow]")
            return

        table = Table(show_header=True, header_style="bold blue", expand=True)
        table.add_column("Programme", width=30)
        table.add_column("Enrollment Date", width=20)
        table.add_column("Status", width=15)
        table.add_column("Completion/Drop Date", width=20)

        while current:
            status_color = "green" if current.status == "Completed" else "red" if current.status == "Dropped" else "yellow"
            completion_date = current.completion_date.strftime("%Y-%m-%d") if current.completion_date else "N/A"

            table.add_row(
                current.programme_name,
                current.enrollment_date.strftime("%Y-%m-%d"),
                f"[{status_color}]{current.status}[/{status_color}]",
                completion_date
            )
            current = current.next

        console.print(table)


class EnhancedProgrammeHistory(ProgrammeHistory):
    def __init__(self):
        super().__init__()
        self.completion_rate = 0
        self.avg_duration = 0

    def calculate_metrics(self):
        """Calculate completion metrics"""
        completed = 0
        total_duration = 0
        current = self.head

        while current:
            if current.status == "Completed":
                completed += 1
                duration = (current.completion_date - current.enrollment_date).days
                total_duration += duration
            current = current.next

        self.completion_rate = (completed / self.count) * 100 if self.count else 0
        self.avg_duration = total_duration / completed if completed else 0

    def get_recommendations(self):
        """Generate recommendations based on history"""
        current = self.head
        completed = []
        dropped = []

        while current:
            if current.status == "Completed":
                completed.append(current.programme_name)
            elif current.status == "Dropped":
                dropped.append(current.programme_name)
            current = current.next

        return {
            'repeat_programs': [p for p in completed if completed.count(p) > 1],
            'avoid_programs': dropped,
            'suggest_advanced': self._get_advanced_options(completed)
        }

    def _get_advanced_options(self, completed):
        """Identify advanced programmes based on completed"""
        programme_levels = {
            'beginner': ['Intro to Python', 'Basic Management'],
            'intermediate': ['Data Analysis', 'Project Management'],
            'advanced': ['Machine Learning', 'Advanced Leadership']
        }

        suggestions = []
        for level, programmes in programme_levels.items():
            if any(p in completed for p in programmes) and level != 'advanced':
                suggestions.extend(programme_levels.get(level + 1, []))
        return suggestions


class RealTimeDashboard:
    def __init__(self, system):
        self.system = system
        self.metrics_history = []
        self.max_history = 100
        self.update_interval = 30  # seconds


    def _get_department_stats(self):
        """Calculate department distribution statistics"""
        dept_stats = {}
        for emp in self.system.employees.values():
            dept_stats[emp.department] = dept_stats.get(emp.department, 0) + 1
        return dept_stats

    def _get_program_stats(self):
        """Calculate program popularity statistics"""
        program_stats = {}
        for emp in self.system.employees.values():
            for program in emp.enrolled_programmes:
                program_stats[program] = program_stats.get(program, 0) + 1
        return program_stats

    def _get_completion_rates(self):
        """Calculate program completion rates"""
        total_enrollments = 0
        total_completions = 0

        for emp in self.system.employees.values():
            current = emp.enrollment_history.head
            while current:
                total_enrollments += 1
                if current.status == "Completed":
                    total_completions += 1
                current = current.next

        return {
            "completion_rate": total_completions / total_enrollments if total_enrollments else 0,
            "total_enrollments": total_enrollments,
            "total_completions": total_completions
        }

    def _get_request_stats(self):
        """Calculate request statistics"""
        priority_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        type_counts = {}

        for req in self.system.request_queue.requests:
            priority_counts[req.priority_level] += 1
            type_counts[req.request_type] = type_counts.get(req.request_type, 0) + 1

        return {
            "priority_counts": priority_counts,
            "type_counts": type_counts,
            "total_requests": self.system.request_queue.size()
        }

    def update_metrics(self):
        """Capture comprehensive system metrics"""
        metrics = {
            'timestamp': datetime.now(),
            'total_employees': len(self.system.employees),
            'full_time_employees': sum(1 for e in self.system.employees.values() if e.full_time_status),
            'part_time_employees': sum(1 for e in self.system.employees.values() if not e.full_time_status),
            'active_enrollments': sum(len(e.enrolled_programmes) for e in self.system.employees.values()),
            'department_distribution': self._get_department_stats(),
            'program_popularity': self._get_program_stats(),
            'completion_rates': self._get_completion_rates(),
            'request_stats': self._get_request_stats(),
            'badge_stats': self._get_badge_stats()
        }

        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)
        return metrics

    def _get_badge_stats(self):
        """Calculate badge distribution statistics"""
        badge_counts = {}
        total_badges = 0
        employees_with_badges = 0

        try:
            with shelve.open('badges_db', flag='r') as badges_db:
                for emp_id in self.system.employees:
                    badges = badges_db.get(str(emp_id), [])
                    if badges:  # If employee has any badges
                        employees_with_badges += 1
                        total_badges += len(badges)
                        for badge in badges:
                            badge_counts[badge] = badge_counts.get(badge, 0) + 1
        except:
            pass

        return {
            'total_badges': total_badges,
            'badge_distribution': badge_counts,
            'employees_with_badges': employees_with_badges
        }

    def display_dashboard(self):
        """Display comprehensive dashboard with multiple visualizations"""
        console.print("\n[bold blue]Training Management System Dashboard[/bold blue]")
        console.print("[dim]Last updated: {0}[/dim]".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        # Get current metrics
        metrics = self.update_metrics()

        # Key Metrics Summary
        self._display_summary(metrics)

        # Department Distribution
        self._display_department_chart(metrics)

        # Program Popularity
        self._display_program_chart(metrics)

        # Request Statistics
        self._display_request_stats(metrics)

        # Completion Rates
        self._display_completion_chart(metrics)

        # Badge Statistics
        self._display_badge_stats(metrics)

        # Trends Over Time
        self._display_trends()

    def _display_summary(self, metrics):
        """Display key metrics summary"""
        table = Table(title="Key Metrics", show_header=True, header_style="bold blue",
                     show_lines=True, expand=True)
        table.add_column("Metric", style="bold", width=25)
        table.add_column("Value", width=15)
        table.add_column("Trend", width=15)

        # Calculate trends if we have historical data
        trend_emoji = "➡️"  # Default to neutral
        if len(self.metrics_history) > 1:
            prev_total = self.metrics_history[-2]['total_employees']
            curr_total = metrics['total_employees']
            trend_emoji = "📈" if curr_total > prev_total else "📉" if curr_total < prev_total else "➡️"

        table.add_row("Total Employees", str(metrics['total_employees']), trend_emoji)
        table.add_row("Full-time", str(metrics['full_time_employees']),
                     self._get_trend_emoji('full_time_employees'))
        table.add_row("Part-time", str(metrics['part_time_employees']),
                     self._get_trend_emoji('part_time_employees'))

        # Most popular program
        if metrics['program_popularity']:
            most_popular = max(metrics['program_popularity'].items(), key=lambda x: x[1])
            table.add_row("Most Popular Program",
                         f"{most_popular[0]} ({most_popular[1]})", "➡️")
        else:
            table.add_row("Most Popular Program", "None", "➡️")

        table.add_row("Pending Requests", str(metrics['request_stats']['total_requests']),
                     self._get_trend_emoji('request_stats.total_requests'))

        completion_rate = metrics['completion_rates']['completion_rate'] * 100
        table.add_row("Completion Rate", f"{completion_rate:.1f}%",
                     self._get_trend_emoji('completion_rates.completion_rate'))

        console.print(table)

    def _display_department_chart(self, metrics):
        """Display department distribution as a bar chart"""
        dept_stats = metrics.get('department_distribution', {})

        if not dept_stats:
            console.print("[yellow]No department data available.[/yellow]")
            return

        # Prepare data for chart
        departments = list(dept_stats.keys())
        counts = list(dept_stats.values())

        # Create simple ASCII bar chart
        max_count = max(counts) if counts else 1
        max_dept_len = max(len(d) for d in departments) if departments else 10

        console.print(f"\n{'Department':<{max_dept_len}} | Employees | Bar")
        console.print("-" * (max_dept_len + 20))

        for dept, count in sorted(dept_stats.items(), key=lambda x: x[1], reverse=True):
            bar_length = int((count / max_count) * 30)
            console.print(
                f"{dept:<{max_dept_len}} | {count:^9} | [cyan]{'█' * bar_length}[/cyan]"
            )

    def _display_program_chart(self, metrics):
        """Display program popularity as a bar chart"""
        program_stats = metrics.get('program_popularity', {})

        if not program_stats:
            console.print("[yellow]No program data available.[/yellow]")
            return

        # Get top 10 programs
        top_programs = sorted(program_stats.items(), key=lambda x: x[1], reverse=True)[:10]

        console.print("\n[bold]Top Training Programs[/bold]")

        max_count = max(p[1] for p in top_programs) if top_programs else 1
        max_prog_len = max(len(p[0]) for p in top_programs) if top_programs else 30

        for program, count in top_programs:
            bar_length = int((count / max_count) * 50)
            console.print(
                f"{program[:max_prog_len]:<{max_prog_len}} [green]{'█' * bar_length}[/green] {count}"
            )

    def _display_request_stats(self, metrics):
        """Display request statistics"""
        request_stats = metrics.get('request_stats', {})

        if not request_stats:
            console.print("[yellow]No request data available.[/yellow]")
            return

        # Priority distribution
        console.print("\n[bold]Request Priority Distribution[/bold]")
        for priority, count in sorted(request_stats['priority_counts'].items()):
            console.print(f"Priority {priority}: [yellow]{count}[/yellow] requests")

        # Request types
        console.print("\n[bold]Request Types[/bold]")
        for req_type, count in sorted(request_stats['type_counts'].items()):
            console.print(f"{req_type}: [cyan]{count}[/cyan]")

    def _display_completion_chart(self, metrics):
        """Display completion metrics with visualization"""
        completion_rates = metrics['completion_rates']

        table = Table(title="Completion Metrics", show_header=True,
                     header_style="bold magenta", show_lines=True)
        table.add_column("Metric", style="bold", width=25)
        table.add_column("Value", width=15)
        table.add_column("Status", width=15)

        overall_rate = completion_rates.get('completion_rate', 0) * 100
        status_style = "green" if overall_rate >= 70 else "yellow" if overall_rate >= 50 else "red"

        table.add_row(
            "Overall Completion Rate",
            f"{overall_rate:.1f}%",
            f"[{status_style}]{'Good' if overall_rate >= 70 else 'Fair' if overall_rate >= 50 else 'Needs Improvement'}[/{status_style}]"
        )
        table.add_row("Total Enrollments", str(completion_rates['total_enrollments']), "")
        table.add_row("Total Completions", str(completion_rates['total_completions']), "")

        console.print(table)

        # Completion rate gauge visualization
        console.print("\n[bold]Completion Rate Gauge[/bold]")
        gauge_width = 50
        filled = int((overall_rate / 100) * gauge_width)
        console.print(
            f"[{'green' if overall_rate >= 70 else 'yellow' if overall_rate >= 50 else 'red'}]"
            f"{'█' * filled}{'░' * (gauge_width - filled)}[/{'green' if overall_rate >= 70 else 'yellow' if overall_rate >= 50 else 'red'}] "
            f"{overall_rate:.1f}%"
        )

    def _display_badge_stats(self, metrics):
        """Display badge statistics"""
        badge_stats = metrics.get('badge_stats', {})

        if not badge_stats:
            console.print("[yellow]No badge data available.[/yellow]")
            return

        console.print("\n[bold]Badge Statistics[/bold]")
        console.print(f"Total Badges Awarded: [green]{badge_stats['total_badges']}[/green]")
        console.print(f"Employees With Badges: [cyan]{badge_stats['employees_with_badges']}[/cyan]")

        if badge_stats['badge_distribution']:
            console.print("\n[bold]Most Common Badges[/bold]")
            for badge, count in sorted(badge_stats['badge_distribution'].items(),
                                      key=lambda x: x[1], reverse=True)[:5]:
                console.print(f"{badge}: [yellow]{count}[/yellow]")

    def _display_trends(self):
        """Display metric trends over time"""
        if len(self.metrics_history) < 2:
            console.print("[yellow]Not enough data to show trends yet.[/yellow]")
            return

        # Prepare trend data
        timestamps = [m['timestamp'].strftime("%H:%M") for m in self.metrics_history]
        employees = [m['total_employees'] for m in self.metrics_history]
        enrollments = [m['active_enrollments'] for m in self.metrics_history]
        requests = [m['request_stats']['total_requests'] for m in self.metrics_history]
        completions = [m['completion_rates']['completion_rate'] * 100 for m in self.metrics_history]

        # Create trend visualization
        console.print("\n[bold blue]System Trends Over Time[/bold blue]")

        # Employee growth
        console.print("\n[bold]Employee Growth:[/bold]")
        self._plot_sparkline(employees, timestamps)

        # Enrollment trends
        console.print("\n[bold]Program Enrollment Trends:[/bold]")
        self._plot_sparkline(enrollments, timestamps)

        # Request queue trends
        console.print("\n[bold]Pending Requests Trends:[/bold]")
        self._plot_sparkline(requests, timestamps)

        # Completion rate trends
        console.print("\n[bold]Completion Rate Trends:[/bold]")
        self._plot_sparkline(completions, timestamps)

    def _plot_sparkline(self, data, labels):
        """Helper to display simple sparkline graphs"""
        if not data:
            return

        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val if max_val != min_val else 1

        # Normalize data to 0-10 range
        normalized = [int(10 * (x - min_val) / range_val) for x in data]

        # Sparkline characters
        spark_chars = "▁▂▃▄▅▆▇█"

        # Build sparkline
        sparkline = ''.join([spark_chars[min(v, len(spark_chars) - 1)] for v in normalized])

        # Display with min/max values
        console.print(f"{sparkline}  [dim](Min: {min_val}, Max: {max_val})[/dim]")

    def _get_trend_emoji(self, metric_name):
        """Helper to get trend emoji for a metric"""
        if len(self.metrics_history) < 2:
            return "➡️"

        # Handle nested metrics
        if '.' in metric_name:
            parts = metric_name.split('.')
            prev_val = self.metrics_history[-2]
            curr_val = self.metrics_history[-1]
            for part in parts:
                prev_val = prev_val.get(part, 0)
                curr_val = curr_val.get(part, 0)
        else:
            prev_val = self.metrics_history[-2].get(metric_name, 0)
            curr_val = self.metrics_history[-1].get(metric_name, 0)

        return "📈" if curr_val > prev_val else "📉" if curr_val < prev_val else "➡️"


class EnhancedRealTimeDashboard(RealTimeDashboard):
    def __init__(self, system):
        super().__init__(system)
        self.alert_thresholds = {
            'pending_requests': 20,
            'completion_rate': 30,
            'department_utilization': 80
        }
        self.visualization_data = self._prepare_visualization_data()
        self.alerts = []

    def _prepare_visualization_data(self):
        """Prepare structured data for visualizations"""
        data = {
            'employees': [],
            'enrollments': [],
            'completions': [],
            'departments': [],
            'requests': []
        }

        # Collect employee data
        for emp in self.system.employees.values():
            data['employees'].append({
                'id': emp.employee_id,
                'department': emp.department,
                'status': 'Full-time' if emp.full_time_status else 'Part-time',
                'program_count': len(emp.enrolled_programmes)
            })

            # Collect enrollment data
            for prog in emp.enrolled_programmes:
                data['enrollments'].append({
                    'employee_id': emp.employee_id,
                    'program': prog,
                    'department': emp.department
                })

            # Collect completion data
            current = emp.enrollment_history.head
            while current:
                if current.status == "Completed":
                    data['completions'].append({
                        'employee_id': emp.employee_id,
                        'program': current.programme_name,
                        'duration': (current.completion_date - current.enrollment_date).days
                    })
                current = current.next

        # Collect department data
        dept_stats = {}
        for emp in self.system.employees.values():
            dept_stats[emp.department] = dept_stats.get(emp.department, 0) + 1
        data['departments'] = [{'name': dept, 'count': count} for dept, count in dept_stats.items()]

        return data

    def _display_employee_metrics(self):
        """Display employee-specific metrics"""
        console.print("\n[bold]Employee Metrics[/bold]")

        # Department distribution
        dept_counts = {}
        status_counts = {'Full-time': 0, 'Part-time': 0}
        program_counts = {}

        for emp in self.system.employees.values():
            dept_counts[emp.department] = dept_counts.get(emp.department, 0) + 1
            status_counts['Full-time' if emp.full_time_status else 'Part-time'] += 1

            for program in emp.enrolled_programmes:
                program_counts[program] = program_counts.get(program, 0) + 1

        # Department pie chart
        console.print("\n[bold]Department Distribution[/bold]")
        dept_data = pd.DataFrame({
            'Department': list(dept_counts.keys()),
            'Count': list(dept_counts.values())
        })
        fig = px.pie(dept_data, names='Department', values='Count', title='Employees by Department')
        fig.show()

        # Status bar chart
        console.print("\n[bold]Employment Status[/bold]")
        status_data = pd.DataFrame({
            'Status': list(status_counts.keys()),
            'Count': list(status_counts.values())
        })
        fig = px.bar(status_data, x='Status', y='Count', title='Full-time vs Part-time')
        fig.show()

        # Program enrollment treemap
        if program_counts:
            console.print("\n[bold]Program Enrollment[/bold]")
            program_data = pd.DataFrame({
                'Program': list(program_counts.keys()),
                'Enrollments': list(program_counts.values())
            })
            fig = px.treemap(program_data, path=['Program'], values='Enrollments',
                             title='Program Enrollment Distribution')
            fig.show()

    def show_dashboard(self):
        """Display a comprehensive dashboard with system statistics"""
        console.print("\n[bold blue]System Dashboard[/bold blue]")
        console.print("[dim]Overview of training management system metrics[/dim]")

        # Update metrics first
        metrics = self.real_time_dashboard.update_metrics()

        # Create dashboard table
        dashboard = Table(show_header=True, header_style="bold blue", show_lines=True)
        dashboard.add_column("Metric", style="bold", width=30)
        dashboard.add_column("Value", style="green", width=20)

        # Employee statistics
        total_employees = metrics['total_employees']
        full_time = sum(1 for e in self.employees.values() if e.full_time_status)
        part_time = total_employees - full_time

        dashboard.add_row("Total Employees", str(total_employees))
        dashboard.add_row("Full-time Employees", str(full_time))
        dashboard.add_row("Part-time Employees", str(part_time))

        # Program statistics
        if metrics['program_popularity']:
            most_common_program, enrollments = max(
                metrics['program_popularity'].items(),
                key=lambda x: x[1],
                default=("None", 0)
            )
            dashboard.add_row("Most Popular Program",
                              f"{most_common_program} ({enrollments} enrollments)")
        else:
            dashboard.add_row("Most Popular Program", "No programs")

        # Request statistics
        pending_requests = metrics['pending_requests']
        dashboard.add_row("Pending Requests", str(pending_requests))

        # Department distribution
        dept_stats = metrics['department_distribution']
        if dept_stats:
            largest_dept, dept_count = max(dept_stats.items(), key=lambda x: x[1])
            dashboard.add_row("Largest Department",
                              f"{largest_dept} ({dept_count} employees)")
        else:
            dashboard.add_row("Largest Department", "No data")

        # Completion rates
        completion_rate = metrics['completion_rates'].get('completion_rate', 0) * 100
        dashboard.add_row("Average Completion Rate", f"{completion_rate:.1f}%")

        console.print(dashboard)

        # Visualizations
        console.print("\n[bold]Department Distribution[/bold]")
        self._display_department_chart()

        # Recent activity
        self._show_recent_activity()

    def show_enhanced_dashboard(self):
        """Display the enhanced dashboard"""
        if not hasattr(self, 'real_time_dashboard'):
            self.real_time_dashboard = RealTimeDashboard(self)
        self.real_time_dashboard.display_dashboard()

    def get_success_prediction(self, employee, program):
        """Get success prediction for employee-program combination"""
        return self.success_predictor.predict_success(employee, program)

    def update_badges(self, employee):
        """Update badges for an employee"""
        return self.badge_system.award_badges(employee)

    def _show_recent_activity(self):
        """Display recent system activity from logs"""
        console.print("\n[bold]Recent Activity[/bold]")
        try:
            with open('logs/employee.log.txt', 'r') as f:
                lines = f.readlines()[-5:]  # Get last 5 log entries
                for line in lines:
                    console.print(f"[dim]{line.strip()}[/dim]")
        except FileNotFoundError:
            console.print("[yellow]No recent activity found.[/yellow]")


    def _display_key_metrics(self, metrics):
        """Display key metrics in a clean table format"""
        table = Table(title="Key Metrics", show_header=True, header_style="bold magenta",
                      show_lines=True, expand=True)
        table.add_column("Metric", style="bold", width=25)
        table.add_column("Value", style="green", width=15)
        table.add_column("Trend", width=15)

        # Calculate trends if we have historical data
        trend_emoji = "➡️"  # Default to neutral
        if len(self.metrics_history) > 1:
            prev_total = self.metrics_history[-2]['total_employees']
            curr_total = metrics['total_employees']
            trend_emoji = "📈" if curr_total > prev_total else "📉" if curr_total < prev_total else "➡️"

        table.add_row("Total Employees", str(metrics['total_employees']), trend_emoji)
        table.add_row("Active Enrollments", str(metrics['active_enrollments']),
                      self._get_trend_emoji('active_enrollments'))
        table.add_row("Pending Requests", str(metrics['pending_requests']),
                      self._get_trend_emoji('pending_requests'))
        table.add_row("Avg Completion Rate", f"{metrics['completion_rates'].get('completion_rate', 0) * 100:.1f}%",
                      self._get_trend_emoji('completion_rate'))

        console.print(table)

    def _display_department_chart(self):
        """Display department distribution as a bar chart"""
        metrics = self.real_time_dashboard.update_metrics()
        dept_stats = metrics.get('department_distribution', {})

        if not dept_stats:
            console.print("[yellow]No department data available.[/yellow]")
            return

        # Prepare data for chart
        departments = list(dept_stats.keys())
        counts = list(dept_stats.values())

        # Create simple ASCII bar chart
        max_count = max(counts) if counts else 1
        max_dept_len = max(len(d) for d in departments) if departments else 10

        console.print(f"\n{'Department':<{max_dept_len}} | Employees | Bar")
        console.print("-" * (max_dept_len + 20))

        for dept, count in sorted(dept_stats.items(), key=lambda x: x[1], reverse=True):
            bar_length = int((count / max_count) * 30)
            console.print(
                f"{dept:<{max_dept_len}} | {count:^9} | [cyan]{'█' * bar_length}[/cyan]"
            )

    def _get_trend_emoji(self, metric_name):
        """Helper to get trend emoji for a metric"""
        if len(self.metrics_history) < 2:
            return "➡️"

        prev_val = self.metrics_history[-2].get(metric_name,
                                                self.metrics_history[-2]['completion_rates'].get(metric_name, 0))
        curr_val = self.metrics_history[-1].get(metric_name,
                                                self.metrics_history[-1]['completion_rates'].get(metric_name, 0))

        if isinstance(prev_val, dict):  # Handle nested metrics
            prev_val = prev_val.get('completion_rate', 0)
            curr_val = curr_val.get('completion_rate', 0)

        return "📈" if curr_val > prev_val else "📉" if curr_val < prev_val else "➡️"

    def _display_department_metrics(self):
        """Display department statistics with bar chart"""
        dept_stats = self._get_department_stats()
        if not dept_stats:
            console.print("[yellow]No department data available.[/yellow]")
            return

        # Create table with department stats
        table = Table(title="Department Breakdown", show_header=True,
                      header_style="bold magenta", show_lines=True)
        table.add_column("Department", style="bold", width=20)
        table.add_column("Employees", width=10)
        table.add_column("Enrollments", width=12)
        table.add_column("Participation", width=15)

        total_employees = sum(dept_stats.values())
        for dept, count in sorted(dept_stats.items(), key=lambda x: x[1], reverse=True):
            participation = (count / total_employees) * 100 if total_employees else 0
            table.add_row(
                dept,
                str(count),
                str(self._get_dept_enrollments(dept)),
                f"{participation:.1f}%"
            )

        console.print(table)

        # Simple bar chart visualization
        console.print("\n[bold]Department Size[/bold]")
        max_count = max(dept_stats.values()) if dept_stats else 1
        for dept, count in sorted(dept_stats.items(), key=lambda x: x[1], reverse=True):
            bar_length = int((count / max_count) * 50)
            console.print(
                f"{dept[:15]:<15} [cyan]{'█' * bar_length}[/cyan] {count}"
            )

    def _get_dept_enrollments(self, department):
        """Count enrollments for a department"""
        count = 0
        for emp in self.system.employees.values():
            if emp.department == department:
                count += len(emp.enrolled_programmes)
        return count

    def _display_program_analytics(self):
        """Display program statistics with visualizations"""
        program_stats = self._get_program_stats()
        if not program_stats:
            console.print("[yellow]No program data available.[/yellow]")
            return

        # Get top 5 programs
        top_programs = sorted(program_stats.items(), key=lambda x: x[1], reverse=True)[:5]

        table = Table(title="Top Training Programs", show_header=True,
                      header_style="bold magenta", show_lines=True)
        table.add_column("Program", style="bold", width=30)
        table.add_column("Enrollments", width=12)
        table.add_column("Success Rate", width=15)

        for program, count in top_programs:
            success_rate = self._get_program_success_rate(program)
            table.add_row(
                program,
                str(count),
                f"{success_rate:.1f}%" if success_rate is not None else "N/A"
            )

        console.print(table)

        # Simple popularity chart
        console.print("\n[bold]Program Popularity[/bold]")
        max_count = max(program_stats.values()) if program_stats else 1
        for program, count in top_programs:
            bar_length = int((count / max_count) * 50)
            console.print(
                f"{program[:25]:<25} [green]{'█' * bar_length}[/green] {count}"
            )

    def _get_program_success_rate(self, program_name):
        """Calculate success rate for a program"""
        completed = 0
        enrolled = 0

        for emp in self.system.employees.values():
            current = emp.enrollment_history.head
            while current:
                if current.programme_name.lower() == program_name.lower():
                    enrolled += 1
                    if current.status == "Completed":
                        completed += 1
                current = current.next

        return (completed / enrolled * 100) if enrolled else None

    def _display_completion_metrics(self):
        """Display completion metrics with visualization"""
        metrics = self.update_metrics()
        completion_rates = metrics['completion_rates']

        table = Table(title="Completion Metrics", show_header=True,
                      header_style="bold magenta", show_lines=True)
        table.add_column("Metric", style="bold", width=25)
        table.add_column("Value", width=15)
        table.add_column("Status", width=15)

        overall_rate = completion_rates.get('completion_rate', 0) * 100
        status_style = "green" if overall_rate >= 70 else "yellow" if overall_rate >= 50 else "red"

        table.add_row(
            "Overall Completion Rate",
            f"{overall_rate:.1f}%",
            f"[{status_style}]{'Good' if overall_rate >= 70 else 'Fair' if overall_rate >= 50 else 'Needs Improvement'}[/{status_style}]"
        )

        # Add department-specific completion rates if available
        dept_rates = self._get_dept_completion_rates()
        if dept_rates:
            best_dept = max(dept_rates.items(), key=lambda x: x[1])
            worst_dept = min(dept_rates.items(), key=lambda x: x[1])

            table.add_row(
                "Highest Performing Dept",
                f"{best_dept[1]:.1f}%",
                f"[green]{best_dept[0]}[/green]"
            )
            table.add_row(
                "Lowest Performing Dept",
                f"{worst_dept[1]:.1f}%",
                f"[red]{worst_dept[0]}[/red]"
            )

        console.print(table)

        # Completion rate gauge visualization
        console.print("\n[bold]Completion Rate[/bold]")
        gauge_width = 50
        filled = int((overall_rate / 100) * gauge_width)
        console.print(
            f"[{'green' if overall_rate >= 70 else 'yellow' if overall_rate >= 50 else 'red'}]"
            f"{'█' * filled}{'░' * (gauge_width - filled)}[/{'green' if overall_rate >= 70 else 'yellow' if overall_rate >= 50 else 'red'}] "
            f"{overall_rate:.1f}%"
        )

    def _get_dept_completion_rates(self):
        """Calculate completion rates by department"""
        dept_stats = {}

        for emp in self.system.employees.values():
            if emp.department not in dept_stats:
                dept_stats[emp.department] = {'completed': 0, 'total': 0}

            current = emp.enrollment_history.head
            while current:
                dept_stats[emp.department]['total'] += 1
                if current.status == "Completed":
                    dept_stats[emp.department]['completed'] += 1
                current = current.next

        return {dept: (stats['completed'] / stats['total'] * 100) if stats['total'] else 0
                for dept, stats in dept_stats.items()}

    def _display_recent_activity(self):
        """Display recent system activity"""
        console.print("\n[bold]Recent Activity[/bold]")

        # Get recent log entries
        try:
            with open('logs/employee.log.txt', 'r') as f:
                lines = f.readlines()[-5:]  # Get last 5 entries

                if not lines:
                    console.print("[dim]No recent activity found[/dim]")
                    return

                table = Table(show_header=False, show_lines=True)
                for line in lines:
                    parts = line.strip().split(' - ')
                    if len(parts) >= 3:
                        timestamp = parts[0]
                        action = ' - '.join(parts[1:])
                        table.add_row(
                            f"[dim]{timestamp}[/dim]",
                            action
                        )
                console.print(table)
        except FileNotFoundError:
            console.print("[yellow]No activity log found[/yellow]")

    def _display_alerts(self):
        """Display system alerts if any"""
        self._check_alerts()
        if not self.alerts:
            return

        console.print("\n[bold red]System Alerts[/bold red]")
        for alert in self.alerts[-3:]:  # Show last 3 alerts
            console.print(f"⚠️ [red]{alert['message']}[/red]")

    def _check_alerts(self):
        """Check for conditions that should trigger alerts"""
        metrics = self.metrics_history[-1] if self.metrics_history else {}

        # Clear old alerts
        self.alerts = [a for a in self.alerts if a['timestamp'] > datetime.now() - timedelta(hours=1)]

        # Check for new alert conditions
        if metrics.get('pending_requests', 0) > self.alert_thresholds['pending_requests']:
            self.alerts.append({
                'type': 'warning',
                'message': f"High pending requests: {metrics['pending_requests']}",
                'timestamp': datetime.now()
            })

        completion_rate = metrics.get('completion_rates', {}).get('completion_rate', 1) * 100
        if completion_rate < self.alert_thresholds['completion_rate']:
            self.alerts.append({
                'type': 'critical',
                'message': f"Low completion rate: {completion_rate:.1f}%",
                'timestamp': datetime.now()
            })


class SuccessPredictor:
    def __init__(self, system):
        self.system = system
        self.model = self._train_model()

    def _train_model(self):
        """Simulated training process"""
        # In a real implementation, this would use ML libraries
        return {"baseline_success_rate": 0.75}

    def predict_success(self, employee, program):
        """Predict program completion likelihood"""
        # Simulated prediction based on employee history
        base_rate = self.model['baseline_success_rate']
        history_factor = len(employee.enrolled_programmes) * 0.02
        return min(0.95, base_rate + history_factor)


class BadgeSystem:
    def __init__(self):
        self.badge_rules = {
            "Fast Learner": self._check_fast_learner,
            "Dedicated": self._check_dedicated,
            "Skill Master": self._check_skill_master
        }

    def _check_fast_learner(self, employee):
        """Check if employee completes programs quickly"""
        completions = [e for e in employee.enrollment_history if e.status == "Completed"]
        if completions:
            avg_duration = sum((c.completion_date - c.enrollment_date).days
                               for c in completions) / len(completions)
            return avg_duration < 14  # Less than 2 weeks average

    def _check_dedicated(self, employee):
        """Check for multiple program enrollments"""
        return len(employee.enrolled_programmes) >= 3

    def _check_skill_master(self, employee):
        """Check for completing a program series"""
        completed = {e.programme_name for e in employee.enrollment_history
                     if e.status == "Completed"}
        return "Advanced Python" in completed and "Python Fundamentals" in completed

    def award_badges(self, employee):
        """Evaluate and award badges to employee"""
        earned_badges = []
        for badge_name, check_func in self.badge_rules.items():
            if check_func(employee):
                earned_badges.append(badge_name)

        if earned_badges:
            with shelve.open('badges_db') as db:
                current = db.get(str(employee.employee_id), [])
                updated = list(set(current + earned_badges))
                db[str(employee.employee_id)] = updated

        return earned_badges


class ProgramAdvisor:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key not configured")

        # Local knowledge base fallback
        self.knowledge_base = {
            "IT": [
                "Python Programming",
                "Data Analysis Fundamentals",
                "Excel"
            ],
            "Marketing": [
                "Leadership 101",
                "Strategic Planning",
                "Conflict Resolution"
            ],
            "Design": [
                "Creative Thinking",
                "Design Principles",
                "Content Creation"
            ]
        }

        try:
            genai.configure(api_key=self.api_key)
            # Use a stable model from your available list
            self.model_name = 'models/gemini-1.5-flash'  # Fast and capable model
            self.model = genai.GenerativeModel(self.model_name)
            self.api_available = True
            system_logger.info(f"Gemini API successfully initialized with model: {self.model_name}")
        except Exception as e:
            console.print(f"[red]Failed to initialize Gemini API: {str(e)}[/red]")
            self.api_available = False
            system_logger.error(f"Gemini API initialization failed: {str(e)}")

    def get_recommendations(self, employee: Employee, query: str = None) -> List[str]:
        """Get recommendations with proper error handling"""
        if not self.api_available:
            return self._get_local_recommendations(employee, query)

        try:
            prompt = self._build_prompt(employee, query)
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 500,
                }
            )
            return self._parse_response(response.text)
        except Exception as e:
            console.print(f"[yellow]AI service error: {str(e)}[/yellow]")
            system_logger.error(f"AI recommendation failed: {str(e)}")
            return self._get_local_recommendations(employee, query)

    def _build_prompt(self, employee: Employee, query: str) -> str:
        """Construct the prompt for Gemini"""
        base_context = f"""
        Employee Profile:
        - Name: {employee.name}
        - Department: {employee.department}
        - Status: {'Full-time' if employee.full_time_status else 'Part-time'}
        - Current Programs: {', '.join(employee.enrolled_programmes) if employee.enrolled_programmes else 'None'}
        """

        if query:
            prompt = base_context + f"""
            The employee asks: "{query}"

            Please provide 3-5 relevant training program recommendations that would help them 
            advance in their career. For each recommendation, include a brief explanation 
            (1 sentence) of why it would be beneficial.

            Format your response with one recommendation per line, like this:
            1. Recommendation 1 - Brief reason
            2. Recommendation 2 - Brief reason
            """
        else:
            prompt = base_context + """
            Please suggest 3-5 training programs that would be most beneficial for this employee
            based on their current role and skills. For each recommendation, include a brief
            explanation (1 sentence) of why it would be beneficial.

            Format your response with one recommendation per line, like this:
            1. Recommendation 1 - Brief reason
            2. Recommendation 2 - Brief reason
            """

        return prompt.strip()

    def _parse_response(self, response: str) -> List[str]:
        """Parse the AI response into a list of recommendations"""
        if not response:
            return []

        # Extract numbered recommendations
        recommendations = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
                # Remove numbering/bullets and any trailing explanation
                rec = line.split('. ')[-1].split(' - ')[0]
                if rec:
                    recommendations.append(rec)

        return recommendations[:5]  # Return max 5 recommendations

    def _get_local_recommendations(self, employee: Employee, query: str = None) -> List[str]:
        """Fallback to local knowledge base recommendations"""
        department = employee.department.lower()
        current_programs = [p.lower() for p in employee.enrolled_programmes]

        # Determine category based on department
        if "IT" in department or "it" in department:
            category = "IT"
        elif "Marketing" in department or "marketing" in department:
            category = "Marketing"
        else:
            category = "Design"

        # Get relevant programs not already enrolled
        recommendations = [
            p for p in self.knowledge_base.get(category, [])
            if p.lower() not in current_programs
        ]

        # If there's a query, sort by relevance
        if query:
            query = query.lower()
            recommendations.sort(
                key=lambda p: self._similarity_score(query, p.lower()),
                reverse=True
            )

        return recommendations[:3]  # Return max 3 local recommendations

    def _similarity_score(self, query: str, text: str) -> float:
        """Simple similarity scoring between query and text"""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        common = query_words & text_words
        return len(common) / len(query_words) if query_words else 0

    def interactive_chat(self, system, employee: 'Employee'):
        """Interactive chat interface for program advice"""
        console.print("\n[bold blue]AI Program Advisor[/bold blue]")

        if self.api_available:
            console.print("[dim]AI-powered advisor (type 'exit' to end)[/dim]\n")
        else:
            console.print("[yellow]Using local knowledge base[/yellow]")
            console.print("[dim]Basic advisor (type 'exit' to end)[/dim]\n")

        while True:
            query = system.get_input("You: ")
            if not query or query.lower() == 'exit':
                break

            recommendations = self.get_recommendations(employee, query)

            if recommendations:
                console.print("\n[bold]Recommended Programs:[/bold]")
                for i, rec in enumerate(recommendations, 1):
                    console.print(f"{i}. {rec}")
            else:
                console.print("\n[yellow]No recommendations found.[/yellow]")


class EnhancedProgramAdvisor(ProgramAdvisor):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self.conversation_history = []
        self.max_history = 10
        self.fallback_responses = [
            "I'm having trouble connecting to the AI service. Here are some general recommendations...",
            "Based on our local knowledge base, you might consider...",
            "The system suggests these programs based on your profile..."
        ]

    def get_recommendations(self, employee: Employee, query: str = None) -> List[str]:
        """Enhanced with conversation history and better fallback"""
        try:
            if not self.api_available:
                raise ConnectionError("AI service unavailable")

            prompt = self._build_prompt(employee, query)

            # Add to conversation history
            self._update_history(f"User: {query}" if query else "User: Requested recommendations")

            model = genai.GenerativeModel(self.model)
            response = model.generate_content(
                {
                    'contents': [{'parts': [{'text': prompt}]}],
                    'generation_config': {
                        'temperature': 0.7,
                        'max_output_tokens': 1000,
                        'top_p': 0.9
                    },
                    'safety_settings': {
                        'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                        'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                        'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                        'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
                    }
                }
            )

            recommendations = self._parse_response(response.text)
            self._update_history(f"AI: Provided {len(recommendations)} recommendations")
            return recommendations

        except Exception as e:
            system_logger.error(f"AI recommendation failed: {str(e)}")
            console.print(f"[yellow]{random.choice(self.fallback_responses)}[/yellow]")
            return self._get_local_recommendations(employee, query)

    def _update_history(self, message: str):
        """Maintain conversation context"""
        if len(self.conversation_history) >= self.max_history:
            self.conversation_history.pop(0)
        self.conversation_history.append(message)

    def _build_prompt(self, employee: Employee, query: str) -> str:
        """Enhanced prompt with context history"""
        base_context = f"""
        Employee Profile:
        - Name: {employee.name}
        - Department: {employee.department}
        - Status: {'Full-time' if employee.full_time_status else 'Part-time'}
        - Current Programs: {', '.join(employee.enrolled_programmes) if employee.enrolled_programmes else 'None'}
        - Completion Rate: {employee.enrollment_history.completion_rate:.1f}%
        """

        history_context = "\n".join(self.conversation_history[-3:]) if self.conversation_history else "No prior context"

        if query:
            prompt = f"""
            {base_context}

            Previous Conversation:
            {history_context}

            New Query: {query}

            Please provide detailed recommendations considering:
            1. Employee's current skill level
            2. Department needs
            3. Career progression paths
            4. Program difficulty based on past completion rates
            """
        else:
            prompt = f"""
            {base_context}

            Previous Conversation:
            {history_context}

            Please suggest programs that would:
            1. Fill skill gaps
            2. Build on existing competencies
            3. Align with department goals
            4. Have appropriate difficulty level
            """

        return prompt.strip()


class FaceRecognizer:
    def __init__(self, system, face_db_file='face_data.dat'):
        self.known_faces = {}
        self.face_db_file = face_db_file
        self.tolerance = 0.5  # Lower is more strict (changed from 0.6)
        self.min_confidence = 0.75  # Increased from 0.7
        self.face_detector = None
        self.face_recognizer = None
        self.model_file = 'openface_nn4.small2.v1.t7'
        self.system = system
        self._initialize_models()
        self._load_face_data()

    def register_face(self, employee_id, image_path=None):
        """Register a face with proper resource cleanup and error handling"""
        video_capture = None  # Initialize outside try block for proper cleanup

        try:
            if employee_id not in self.system.employees:
                console.print("[red]Employee ID not found in system[/red]")
                return False

            if image_path:
                # Load from image file
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not load image from {image_path}")
            else:
                # Capture from webcam
                video_capture = cv2.VideoCapture(0)
                if not video_capture.isOpened():
                    raise RuntimeError("Could not open webcam")

                console.print("\nPlease face the camera for registration...")
                console.print("Press SPACE to capture or ESC to cancel")

                while True:
                    ret, frame = video_capture.read()
                    if not ret:
                        raise RuntimeError("Could not capture frame from webcam")

                    # Display face detection overlay
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_detector.detectMultiScale(gray, 1.1, 5)

                    # Draw rectangle around faces
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    cv2.imshow("Face Registration", frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key == 27:  # ESC
                        console.print("[yellow]Registration cancelled[/yellow]")
                        return False
                    elif key == 32:  # SPACE
                        image = frame.copy()
                        break

            # Get face encodings
            encodings = self._get_face_encodings(image)
            if not encodings:
                console.print("[red]No face detected - please ensure your face is clearly visible[/red]")
                return False
            elif len(encodings) > 1:
                console.print("[red]Multiple faces detected - please register one face at a time[/red]")
                return False

            # Check quality of the face encoding
            encoding = encodings[0]
            if np.linalg.norm(encoding) < 0.1:  # Very weak encoding
                console.print("[red]Poor quality face capture - please try again with better lighting[/red]")
                return False

            # Store the face encoding
            self.known_faces[employee_id] = encoding

            # Save face data
            if not self._save_face_data():
                    raise RuntimeError("Failed to save face data")


            # Update the employee's face_registered flag
            self.system.employees[employee_id].face_registered = True
            self.system.save_data()  # Save the employee data

            console.print(f"[green]Successfully registered face for employee {employee_id}[/green]")
            system_logger.info(f"Registered face for employee {employee_id}")
            return True

        except Exception as e:
            console.print(f"[red]Error during face registration: {str(e)}[/red]")
            system_logger.error(f"Face registration failed: {str(e)}", exc_info=True)
            return False
        finally:
            # Ensure resources are cleaned up in all cases
            if video_capture is not None:
                video_capture.release()
            cv2.destroyAllWindows()

    def _initialize_models(self):
        """Initialize face detection and recognition models with proper error handling"""
        try:
            # Load face detection model
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_path):
                raise FileNotFoundError(f"Haar cascade file not found at {cascade_path}")
            self.face_detector = cv2.CascadeClassifier(cascade_path)

            # Download and load face recognition model if not present
            if not os.path.exists(self.model_file):
                console.print("[yellow]Face recognition model not found, downloading...[/yellow]")
                self._download_model()

            self.face_recognizer = cv2.dnn.readNetFromTorch(self.model_file)
            system_logger.info("Face recognition models loaded successfully")

        except Exception as e:
            system_logger.error(f"Failed to initialize face recognition: {str(e)}")
            console.print(f"[red]Error initializing face recognition: {str(e)}[/red]")
            raise RuntimeError("Face recognition initialization failed")

    def _download_model(self):
        """Download the OpenFace model file with better error handling"""
        model_url = "https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7"
        try:
            console.print("[yellow]Downloading face recognition model... (This may take a while)[/yellow]")

            # Create temp directory if it doesn't exist
            os.makedirs('temp', exist_ok=True)
            temp_file = os.path.join('temp', 'model_download.tmp')

            # Download with progress feedback
            def progress(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\rDownload progress: {percent}%")
                sys.stdout.flush()

            urllib.request.urlretrieve(model_url, temp_file, progress)
            print()  # New line after progress

            # Verify download completed
            if os.path.getsize(temp_file) < 1000000:  # Rough check for minimum size
                raise RuntimeError("Downloaded file seems too small - download may have failed")

            # Move to final location
            shutil.move(temp_file, self.model_file)
            console.print("[green]Model downloaded successfully![/green]")

        except Exception as e:
            console.print(f"[red]Failed to download model: {str(e)}[/red]")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise RuntimeError("Model download failed")

    def _load_face_data(self):
        """Load face encodings from file with better error handling"""
        try:
            if os.path.exists(self.face_db_file):
                with open(self.face_db_file, 'rb') as f:
                    self.known_faces = pickle.load(f)
                system_logger.info(f"Loaded {len(self.known_faces)} face encodings")
            else:
                system_logger.info("No existing face data found, starting fresh")
                self.known_faces = {}

        except Exception as e:
            system_logger.error(f"Could not load face data - {str(e)}")
            console.print(f"[yellow]Warning: Could not load face data - starting fresh[/yellow]")
            self.known_faces = {}

    def _save_face_data(self):
        """Save face encodings to file with atomic write and proper error handling"""
        try:
            # Create directory if it doesn't exist
            db_dir = os.path.dirname(self.face_db_file)
            if db_dir:  # Only create directory if path contains a directory
                os.makedirs(db_dir, exist_ok=True)

            # Write to temp file first
            temp_file = self.face_db_file + '.tmp'
            with open(temp_file, 'wb') as f:
                pickle.dump(self.known_faces, f)
                f.flush()  # Ensure data is written to disk

            # Verify the temp file was written correctly
            if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
                raise RuntimeError("Failed to write face data to temporary file")

            # Replace old file atomically
            if os.path.exists(self.face_db_file):
                os.remove(self.face_db_file)
            os.rename(temp_file, self.face_db_file)

            system_logger.info(f"Saved {len(self.known_faces)} face encodings")
            return True

        except Exception as e:
            system_logger.error(f"Could not save face data - {str(e)}")
            console.print(f"[red]Error saving face data: {str(e)}[/red]")
            if 'temp_file' in locals() and os.path.exists(temp_file):
                os.remove(temp_file)
            return False

    def _get_face_encodings(self, image):
        """Improved face encoding extraction with quality checks"""
        if image is None:
            return []

        try:
            # Convert to RGB and equalize histogram
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            limg = cv2.merge([clahe.apply(l), a, b])
            processed = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

            # Detect faces with stricter parameters
            gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.2,  # Increased from 1.1
                minNeighbors=7,  # Increased from 5
                minSize=(120, 120),  # Increased minimum face size
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            if len(faces) == 0:
                return []

            encodings = []
            for (x, y, w, h) in faces:
                try:
                    # Extract face ROI with padding
                    padding = 0.25  # Increased padding
                    x_pad = int(w * padding)
                    y_pad = int(h * padding)
                    face_roi = processed[
                               max(0, y - y_pad):min(y + h + y_pad, processed.shape[0]),
                               max(0, x - x_pad):min(x + w + x_pad, processed.shape[1])
                               ]

                    # Create blob with improved parameters
                    face_blob = cv2.dnn.blobFromImage(
                        face_roi,
                        1.0 / 255,
                        (96, 96),
                        (0, 0, 0),
                        swapRB=False,
                        crop=False
                    )

                    # Get face encoding
                    self.face_recognizer.setInput(face_blob)
                    encoding = self.face_recognizer.forward()
                    encodings.append(encoding.flatten())

                except Exception as e:
                    continue

            return encodings

        except Exception as e:
            system_logger.error(f"Face encoding failed: {str(e)}")
            return []

    def authenticate(self, employee_id, image_path=None):
        """Authenticate a face with improved matching"""
        try:
            if employee_id not in self.known_faces:
                console.print("[red]No face data found for this employee[/red]")
                return False

            if image_path:
                image = cv2.imread(image_path)
                if image is None:
                    console.print(f"[red]Error loading image from {image_path}[/red]")
                    return False
            else:
                video_capture = cv2.VideoCapture(0)
                if not video_capture.isOpened():
                    console.print("[red]Error: Could not open webcam[/red]")
                    return False

                console.print("\nPlease face the camera for authentication...")
                console.print("Press SPACE to capture or ESC to cancel")

                while True:
                    ret, frame = video_capture.read()
                    if not ret:
                        console.print("[red]Error: Could not capture frame[/red]")
                        return False

                    # Display face detection overlay
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_detector.detectMultiScale(gray, 1.1, 5)

                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    cv2.imshow("Face Authentication", frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key == 27:  # ESC
                        console.print("[yellow]Authentication cancelled[/yellow]")
                        return False
                    elif key == 32:  # SPACE
                        image = frame.copy()
                        break

                video_capture.release()
                cv2.destroyAllWindows()

            # Get face encodings with improved preprocessing
            encodings = self._get_face_encodings(image)
            if not encodings:
                console.print("[red]No face detected[/red]")
                return False

            # Compare with stored encoding using improved matching
            stored_encoding = self.known_faces[employee_id]

            # Convert to numpy arrays if they aren't already
            current_encoding = np.array(encodings[0])
            stored_encoding = np.array(stored_encoding)

            # Normalize the encodings
            current_encoding = current_encoding / np.linalg.norm(current_encoding)
            stored_encoding = stored_encoding / np.linalg.norm(stored_encoding)

            # Calculate cosine similarity (better than Euclidean for face recognition)
            similarity = np.dot(current_encoding, stored_encoding)

            # Debug output
            console.print(f"[dim]Similarity score: {similarity:.4f}[/dim]")

            # Check against both tolerance and minimum confidence
            if similarity >= (1 - self.tolerance) and similarity >= self.min_confidence:
                console.print(f"[green]Authentication successful! (similarity: {similarity:.4f})[/green]")
                return True

            console.print(f"[red]Authentication failed (similarity: {similarity:.4f})[/red]")
            return False

        except Exception as e:
            console.print(f"[red]Error during authentication: {str(e)}[/red]")
            return False


class TrainingManagementSystem:
    def __init__(self):
        self.employees = {}
        self.logged_in_user = None
        self.admin_username = "admin"
        self.admin_password = "admin123"
        self.request_queue = EmployeeRequestQueue()
        self.request_history = RequestHistory()
        self.department_tree = self._build_department_tree()
        self.load_data()
        self.real_time_dashboard = RealTimeDashboard(self)  # Moved this line up
        self.program_advisor = ProgramAdvisor(os.getenv("GEMINI_API_KEY"))
        system_logger.info("System initialized")
        self.success_predictor = SuccessPredictor(self)
        self.badge_system = BadgeSystem()
        self.biometric_auth = FaceRecognizer(self)
        self.stop_dashboard = False
        self.lock = threading.Lock()
        # Encryption setup
        self.encryption_key = self._load_or_generate_key()
        self.encryptor = Fernet(self.encryption_key)

    def _load_or_generate_key(self):
        """Load encryption key or generate a new one"""
        key_file = 'encryption.key'
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            return key

    def encrypt_sensitive_data(self, text):
        """Encrypt text only if it's not already encrypted"""
        if not text:
            return ""

        # Don't encrypt if it's already encrypted (Fernet tokens always end with ==)
        if isinstance(text, str) and text.endswith("==") and len(text) > 50:
            return text

        if isinstance(text, str):
            text = text.encode('utf-8')

        return self.encryptor.encrypt(text).decode('utf-8')

    def decrypt_sensitive_data(self, encrypted_text):
        """Always try to decrypt sensitive text; return original if it fails."""
        if not encrypted_text:
            return ""

        try:
            if isinstance(encrypted_text, str):
                encrypted_text_bytes = encrypted_text.encode("utf-8")
            else:
                encrypted_text_bytes = encrypted_text

            decrypted = self.encryptor.decrypt(encrypted_text_bytes).decode("utf-8")
            return decrypted
        except Exception as e:
            system_logger.error(f"Decryption failed for value: {encrypted_text[:20]}... | Error: {str(e)}")
            return encrypted_text if isinstance(encrypted_text, str) else encrypted_text.decode("utf-8")

    def register_biometric(self, employee_id):
        """Register an employee's face for biometric authentication"""
        console.print("\n[bold blue]Biometric Registration[/bold blue]")

        if employee_id not in self.employees:
            console.print("[red]Employee ID not found.[/red]")
            return False

        employee = self.employees[employee_id]
        video_capture = None  # Initialize outside try block for proper cleanup

        console.print(f"Registering face for employee {employee_id} ({employee.name})")
        console.print("\nPlease look directly at the camera and ensure good lighting.")
        console.print("Make sure your face is clearly visible and well-lit.")

        try:
            # Give 3 attempts
            max_attempts = 3
            for attempt in range(max_attempts):
                console.print(f"\nAttempt {attempt + 1} of {max_attempts}")
                success = self.biometric_auth.register_face(employee_id)
                if success:
                    employee.face_registered = True
                    self.save_data()
                    console.print(f"\n[green]Successfully registered face for employee {employee_id}[/green]")
                    return True
                else:
                    if attempt < max_attempts - 1:
                        console.print(f"[yellow]Registration attempt failed. Try again...[/yellow]")

            console.print("[red]Maximum registration attempts reached.[/red]")
            return False

        except Exception as e:
            console.print(f"[red]Error during face registration: {str(e)}[/red]")
            system_logger.error(f"Face registration failed: {str(e)}", exc_info=True)
            return False
        finally:
            # Ensure resources are cleaned up in all cases
            if video_capture is not None:
                video_capture.release()
            cv2.destroyAllWindows()

    # In the biometric_login method:
    def biometric_login(self):
        console.print("\n[bold blue]Biometric Login[/bold blue]")
        try:
            employee_id = self.get_input("Enter your employee ID: ", required=True)
            if not employee_id:
                return False

            employee_id = int(employee_id)
            if employee_id not in self.employees:
                console.print("[red]Employee ID not found.[/red]")
                return False

            employee = self.employees[employee_id]

            # Check if face is registered in both places
            if (not employee.face_registered or
                    employee_id not in self.biometric_auth.known_faces):
                console.print("[yellow]No face registered for this employee.[/yellow]")
                register = self.get_input("Would you like to register your face now? (Y/N): ")
                if register and register.upper() == 'Y':
                    if self.register_biometric(employee_id):
                        console.print("[green]Face registration successful! Please login again.[/green]")
                    return False
                else:
                    return False

            # Proceed with authentication
            console.print("\n[bold]Please look directly at the camera for authentication...[/bold]")
            if self.biometric_auth.authenticate(employee_id):
                self.logged_in_user = employee_id
                console.print(f"\n[green]Welcome {employee.name}![/green]")
                system_logger.info(f"Employee {employee_id} logged in via biometric authentication")
                return True

            console.print("[red]Authentication failed.[/red]")
            return False

        except Exception as e:
            console.print(f"[red]Error during biometric login: {str(e)}[/red]")
            system_logger.error(f"Biometric login failed: {str(e)}", exc_info=True)
            return False

    def login(self):
        console.print("\n[bold blue]Login[/bold blue]")
        console.print("1. Admin\n2. Employee\n3. Biometric Login\n4. Exit")

        while True:
            choice = self.get_input("Enter choice (1-4): ", required=True)
            if choice is None:
                continue

            if choice == '1':
                # Admin login
                username = self.get_input("Username: ", required=True)
                password = self.get_input("Password: ", required=True)

                if username == self.admin_username and password == self.admin_password:
                    self.logged_in_user = "admin"
                    action_logger.info("Admin logged in")
                    console.print("\n[green]Admin login successful![/green]")
                    return True
                else:
                    console.print("[red]Invalid credentials.[/red]")
                    return False

            elif choice == '2':
                # Employee password login
                employee_id = self.select_employee()
                if not employee_id:
                    console.print("[red]Invalid Employee ID.[/red]")
                    return False

                employee = self.employees[employee_id]

                if employee.password is None:
                    console.print("\n[yellow]First-time login. Set your password.[/yellow]")
                    while True:
                        password = self.get_input("Set password: ", required=True)
                        confirm = self.get_input("Confirm password: ", required=True)

                        if password == confirm:
                            hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                            employee.password = hashed.decode('utf-8')
                            self.save_data()
                            self.logged_in_user = employee_id
                            action_logger.info(f"Employee {employee_id} set password")
                            console.print("\n[green]Password set and login successful![/green]")
                            return True
                        else:
                            console.print("[red]Passwords do not match. Try again.[/red]")
                else:
                    password = self.get_input("Enter password: ", required=True)
                    if bcrypt.checkpw(password.encode('utf-8'), employee.password.encode('utf-8')):
                        self.logged_in_user = employee_id
                        action_logger.info(f"Employee {employee_id} logged in")
                        console.print("\n[green]Login successful![/green]")
                        return True
                    else:
                        console.print("[red]Incorrect password.[/red]")
                        return False

            elif choice == '3':
                # Biometric login
                if self.biometric_login():
                    return True
                else:
                    continue

            elif choice == '4':
                self.save_data()
                console.print("\n[blue]Goodbye![/blue]")
                sys.exit(0)
            else:
                console.print("[red]Invalid choice.[/red]")
                continue

    def _build_department_tree(self):
        """Build department hierarchy tree"""
        root = DepartmentTreeNode("Organization")

        # Sample department structure - could be loaded from config
        hr = root.add_child("Human Resources")
        hr.add_child("Recruitment")
        hr.add_child("Employee Development")

        it = root.add_child("Information Technology")
        it.add_child("Software Development")
        it.add_child("Infrastructure")
        it.add_child("Data Science")

        finance = root.add_child("Finance")
        finance.add_child("Accounting")
        finance.add_child("Financial Planning")

        # Assign employees to departments
        for emp in self.employees.values():
            dept_node = root.find_department(emp.department)
            if dept_node:
                dept_node.add_employee(emp)
            else:
                # Auto-create department if not found
                new_dept = root.add_child(emp.department)
                new_dept.add_employee(emp)

        return root

    def save_data(self):
        """Save employee data and request queue to disk"""
        try:
            os.makedirs('data', exist_ok=True)
            db_path = os.path.join('data', 'employee_data')

            with shelve.open(db_path) as db:
                # Save employee data
                employee_data = {}
                for eid, emp in self.employees.items():
                    # Convert enrollment history to a serializable format
                    history_data = []
                    current = emp.enrollment_history.head
                    while current:
                        history_data.append({
                            'programme_name': current.programme_name,
                            'enrollment_date': current.enrollment_date,
                            'status': current.status,
                            'completion_date': current.completion_date,
                            'feedback': current.feedback,
                            'rating': current.rating
                        })
                        current = current.next

                    emp_dict = emp.to_dict()
                    emp_dict['enrollment_history'] = history_data
                    employee_data[eid] = emp_dict

                db['employees'] = employee_data

                # Save request queue
                db['request_queue'] = [req.__dict__ for req in self.request_queue.requests]
                db['processed_requests'] = [req.__dict__ for req in self.request_queue.processed_requests]

        except Exception as e:
            console.print(f"[red]Error saving data: {str(e)}[/red]")
            system_logger.error(f"Data save failed: {str(e)}")

    def load_data(self):
        """Load employee data and request queue from disk"""
        try:
            db_path = os.path.join('data', 'employee_data')
            if not os.path.exists(f'{db_path}.dat'):
                return

            with shelve.open(db_path, flag='r') as db:
                # Load employee data
                employee_data = db.get('employees', {})
                for eid, emp_data in employee_data.items():
                    employee = Employee(
                        name=emp_data['name'],
                        employee_id=int(eid),
                        email=emp_data['email'],
                        department=emp_data['department'],
                        full_time_status=emp_data['full_time_status'],
                        enrolled_programmes=emp_data['enrolled_programmes']
                    )
                    employee.password = emp_data.get('password')

                    # Restore enrollment history
                    if 'enrollment_history' in emp_data:
                        history = ProgrammeHistory()
                        for item in emp_data['enrollment_history']:
                            node = ProgrammeEnrollmentNode(
                                item['programme_name'],
                                item['enrollment_date'],
                                item['status']
                            )
                            node.completion_date = item['completion_date']
                            node.feedback = item['feedback']
                            node.rating = item['rating']

                            if not history.head:
                                history.head = node
                                history.tail = node
                            else:
                                history.tail.next = node
                                history.tail = node
                            history.count += 1

                        employee.enrollment_history = history

                    self.employees[int(eid)] = employee

                # Load request queue
                if 'request_queue' in db:
                    for req_dict in db['request_queue']:
                        req = EmployeeRequest(
                            req_dict['employee_id'],
                            req_dict['request_type'],
                            req_dict['priority_level'],
                            req_dict['request_details']
                        )
                        req.timestamp = req_dict['timestamp']
                        self.request_queue.add_request(req)

                if 'processed_requests' in db:
                    for req_dict in db['processed_requests']:
                        req = EmployeeRequest(
                            req_dict['employee_id'],
                            req_dict['request_type'],
                            req_dict['priority_level'],
                            req_dict['request_details']
                        )
                        req.timestamp = req_dict['timestamp']
                        self.request_queue.processed_requests.append(req)

        except Exception as e:
            console.print(f"[red]Error loading data: {str(e)}[/red]")
            system_logger.error(f"Data load failed: {str(e)}")

    # Quick Sort Implementation
    def quick_sort_employees(self, employees):
        """Quick sort implementation for employees by department (ascending) and name (ascending)"""
        if len(employees) <= 1:
            return employees

        pivot = employees[len(employees) // 2]
        left = [emp for emp in employees if
                (emp.department < pivot.department) or
                (emp.department == pivot.department and emp.name < pivot.name)]
        middle = [emp for emp in employees if
                  emp.department == pivot.department and emp.name == pivot.name]
        right = [emp for emp in employees if
                 (emp.department > pivot.department) or
                 (emp.department == pivot.department and emp.name > pivot.name)]

        return self.quick_sort_employees(left) + middle + self.quick_sort_employees(right)

    def display_sorted_by_department(self):
        """Display employees sorted by department and name using quick sort"""
        console.print("\n[bold blue]Employees Sorted by Department and Name (Quick Sort)[/bold blue]")

        # Get all employees and sort them
        employee_list = list(self.employees.values())
        sorted_employees = self.quick_sort_employees(employee_list)

        # Display results in tabular format
        table = Table(show_header=True, header_style="bold blue", show_lines=True)
        table.add_column("ID", style="dim", width=10)
        table.add_column("Name", width=20)
        table.add_column("Department", width=20)
        table.add_column("Status", width=12)
        table.add_column("Programmes", width=30)

        for emp in sorted_employees:
            status = "Full-time" if emp.full_time_status else "Part-time"
            programmes = ", ".join(emp.enrolled_programmes) if emp.enrolled_programmes else "None"
            table.add_row(
                str(emp.employee_id),
                emp.name,
                emp.department,
                status,
                programmes
            )

        console.print(table)
        action_logger.info("Displayed employees sorted by department and name using quick sort")

    # Merge Sort Implementation
    def merge_sort_employees(self, employees):
        """Merge sort implementation for employees by num of programmes and ID"""
        if len(employees) <= 1:
            return employees

        mid = len(employees) // 2
        left = self.merge_sort_employees(employees[:mid])
        right = self.merge_sort_employees(employees[mid:])

        return self.merge(left, right)

    def merge(self, left, right):
        """Merge helper function for merge sort"""
        result = []
        i = j = 0

        while i < len(left) and j < len(right):
            # Sort by number of programmes (ascending), then by employee ID (ascending)
            if len(left[i].enrolled_programmes) < len(right[j].enrolled_programmes):
                result.append(left[i])
                i += 1
            elif len(left[i].enrolled_programmes) > len(right[j].enrolled_programmes):
                result.append(right[j])
                j += 1
            else:  # If same number of programmes, sort by ID
                if left[i].employee_id < right[j].employee_id:
                    result.append(left[i])
                    i += 1
                else:
                    result.append(right[j])
                    j += 1

        result.extend(left[i:])
        result.extend(right[j:])
        return result

    def display_sorted_by_programmes(self):
        """Display employees sorted by number of programmes and ID using merge sort"""
        console.print("\n[bold blue]Employees Sorted by Programmes and ID (Merge Sort)[/bold blue]")

        # Get all employees and sort them
        employee_list = list(self.employees.values())
        sorted_employees = self.merge_sort_employees(employee_list)

        # Display results in tabular format
        table = Table(show_header=True, header_style="bold blue", show_lines=True)
        table.add_column("ID", style="dim", width=10)
        table.add_column("Name", width=20)
        table.add_column("Department", width=20)
        table.add_column("# Programmes", width=12)
        table.add_column("Programmes", width=30)

        for emp in sorted_employees:
            table.add_row(
                str(emp.employee_id),
                emp.name,
                emp.department,
                str(len(emp.enrolled_programmes)),
                ", ".join(emp.enrolled_programmes) if emp.enrolled_programmes else "None"
            )

        console.print(table)
        action_logger.info("Displayed employees sorted by programmes and ID using merge sort")

        # Add department filter option
        self.filter_by_department(sorted_employees)

    def filter_by_department(self, employees):
        """Filter the sorted list by department"""
        departments = sorted(set(emp.department for emp in employees))

        console.print("\n[bold]Available Departments:[/bold]")
        for i, dept in enumerate(departments, 1):
            console.print(f"{i}. {dept}")

        choice = self.get_input("\nSelect department to filter (number) or press Enter to skip: ")
        if not choice:
            return

        try:
            dept_idx = int(choice) - 1
            if 0 <= dept_idx < len(departments):
                selected_dept = departments[dept_idx]
                filtered = [emp for emp in employees if emp.department == selected_dept]

                console.print(f"\n[bold]{selected_dept} Department Employees:[/bold]")

                table = Table(show_header=True, header_style="bold blue", show_lines=True)
                table.add_column("ID", style="dim", width=10)
                table.add_column("Name", width=20)
                table.add_column("# Programmes", width=12)
                table.add_column("Programmes", width=30)

                for emp in filtered:
                    table.add_row(
                        str(emp.employee_id),
                        emp.name,
                        str(len(emp.enrolled_programmes)),
                        ", ".join(emp.enrolled_programmes) if emp.enrolled_programmes else "None"
                    )

                console.print(table)
                action_logger.info(f"Filtered employees by department: {selected_dept}")
            else:
                console.print("[red]Invalid department selection.[/red]")
        except ValueError:
            console.print("[red]Please enter a valid number.[/red]")

    # Employee Request Management
    def validate_employee_id_for_request(self, employee_id):
        """Validate employee ID exists using binary search"""
        sorted_ids = sorted(self.employees.keys())
        low = 0
        high = len(sorted_ids) - 1

        while low <= high:
            mid = (low + high) // 2
            if sorted_ids[mid] == employee_id:
                return True
            elif sorted_ids[mid] < employee_id:
                low = mid + 1
            else:
                high = mid - 1

        return False

    def add_employee_request(self):
        """Add a new employee request to the priority queue"""
        console.print("\n[bold blue]Add Employee Request[/bold blue]")

        # Employee ID validation
        while True:
            employee_id = self.get_input("Enter employee ID (or 'back' to cancel): ", required=True)
            if employee_id is None:
                return

            try:
                employee_id = int(employee_id)

                # Binary search validation
                if not self.validate_employee_id_for_request(employee_id):
                    console.print("[red]Employee ID not found.[/red]")
                    continue

                # If ID exists, prompt for confirmation if duplicate requests exist
                if any(req.employee_id == employee_id for req in self.request_queue.requests):
                    confirm = self.get_input(
                        f"Employee {employee_id} already has pending requests. Continue? (Y/N): ",
                        required=True
                    )
                    if confirm and confirm.upper() != 'Y':
                        continue

                break
            except ValueError:
                console.print("[red]Please enter a valid ID.[/red]")

        # Request type
        request_types = ["Enrollment", "Feedback", "Information", "Technical", "Other"]
        console.print("\n[bold]Request Types:[/bold]")
        for i, req_type in enumerate(request_types, 1):
            console.print(f"{i}. {req_type}")

        while True:
            type_choice = self.get_input("Select request type (1-5): ", required=True)
            if type_choice is None:
                return
            try:
                type_idx = int(type_choice) - 1
                if 0 <= type_idx < len(request_types):
                    request_type = request_types[type_idx]
                    break
                console.print("[red]Invalid selection.[/red]")
            except ValueError:
                console.print("[red]Please enter a number.[/red]")

        # Priority level
        while True:
            priority = self.get_input("Priority level (1-5, 1=highest): ", required=True)
            if priority is None:
                return
            try:
                priority_level = int(priority)
                if 1 <= priority_level <= 5:
                    break
                console.print("[red]Priority must be between 1-5.[/red]")
            except ValueError:
                console.print("[red]Please enter a number.[/red]")

        # Request details
        details = self.get_input("Request details: ", required=True)
        if details is None:
            return

        # Create and add request
        request = EmployeeRequest(employee_id, request_type, priority_level, details)
        self.request_queue.add_request(request)
        self.request_history.record_action('add', request)

        console.print("\n[green]Request added successfully![/green]")
        action_logger.info(f"Added request for employee {employee_id}: {request_type} (Priority: {priority_level})")

        # Show queue position
        console.print(f"\n[dim]Current queue position: {self.request_queue.size()}[/dim]")

    def view_request_statistics(self):
        """Display queue statistics and filtering options"""
        console.print("\n[bold blue]Employee Request Queue Statistics[/bold blue]")

        total_requests = self.request_queue.size()
        console.print(f"\n[bold]Total Requests in Queue:[/bold] {total_requests}")

        if total_requests == 0:
            return

        # Show priority distribution
        console.print("\n[bold]Requests by Priority:[/bold]")
        priority_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for req in self.request_queue.requests:
            priority_counts[req.priority_level] += 1

        for priority, count in sorted(priority_counts.items()):
            console.print(f"Priority {priority}: {count} request(s)")

        # Show type distribution
        console.print("\n[bold]Requests by Type:[/bold]")
        type_counts = {}
        for req in self.request_queue.requests:
            type_counts[req.request_type] = type_counts.get(req.request_type, 0) + 1

        for req_type, count in sorted(type_counts.items()):
            console.print(f"{req_type}: {count} request(s)")

        # Filtering options
        console.print("\n1. Filter by Request Type\n2. Filter by Priority Level\n3. Back")
        choice = self.get_input("Select option: ")

        if choice == '1':
            request_types = sorted(set(req.request_type for req in self.request_queue.requests))
            console.print("\n[bold]Available Request Types:[/bold]")
            for i, req_type in enumerate(request_types, 1):
                console.print(f"{i}. {req_type}")

            type_choice = self.get_input("Select type to filter (number): ")
            if type_choice:
                try:
                    type_idx = int(type_choice) - 1
                    if 0 <= type_idx < len(request_types):
                        selected_type = request_types[type_idx]
                        filtered = self.request_queue.get_requests_by_type(selected_type)

                        console.print(f"\n[bold]{selected_type} Requests:[/bold]")
                        self.display_requests_table(filtered)
                    else:
                        console.print("[red]Invalid selection.[/red]")
                except ValueError:
                    console.print("[red]Please enter a number.[/red]")

        elif choice == '2':
            priority_choice = self.get_input("Enter priority level to filter (1-5): ")
            if priority_choice:
                try:
                    priority_level = int(priority_choice)
                    if 1 <= priority_level <= 5:
                        filtered = self.request_queue.get_requests_by_priority(priority_level)

                        console.print(f"\n[bold]Priority {priority_level} Requests:[/bold]")
                        self.display_requests_table(filtered)
                    else:
                        console.print("[red]Priority must be 1-5.[/red]")
                except ValueError:
                    console.print("[red]Please enter a number.[/red]")

    def display_requests_table(self, requests):
        """Display requests in a formatted table"""
        if not requests:
            console.print("[yellow]No requests found.[/yellow]")
            return

        # Sort by priority and timestamp
        sorted_requests = sorted(requests, key=lambda x: (x.priority_level, x.timestamp))

        table = Table(show_header=True, header_style="bold blue", show_lines=True)
        table.add_column("Position", style="dim", width=8)
        table.add_column("Employee ID", width=10)
        table.add_column("Type", width=15)
        table.add_column("Priority", width=10)
        table.add_column("Details", width=40)
        table.add_column("Timestamp", width=20)

        for i, req in enumerate(sorted_requests, 1):
            table.add_row(
                str(i),
                str(req.employee_id),
                req.request_type,
                str(req.priority_level),
                req.request_details,
                req.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            )

        console.print(table)

    def process_next_request(self):
        """Process the next request in the priority queue"""
        console.print("\n[bold blue]Process Next Request[/bold blue]")

        if self.request_queue.is_empty():
            console.print("[yellow]No requests in queue.[/yellow]")
            return

        # Peek at next request
        next_request = self.request_queue.peek_next_request()
        self.request_history.record_action('process', next_request)

        console.print("\n[bold]Next Request to Process:[/bold]")

        table = Table(show_header=True, header_style="bold blue", show_lines=True)
        table.add_column("Field", style="dim", width=15)
        table.add_column("Value", width=40)

        employee = self.employees.get(next_request.employee_id, None)
        emp_name = employee.name if employee else "Unknown"

        table.add_row("Employee ID", str(next_request.employee_id))
        table.add_row("Employee Name", emp_name)
        table.add_row("Request Type", next_request.request_type)
        table.add_row("Priority", str(next_request.priority_level))
        table.add_row("Details", next_request.request_details)
        table.add_row("Submitted", next_request.timestamp.strftime("%Y-%m-%d %H:%M:%S"))

        console.print(table)

        # Confirmation
        confirm = self.get_input("\nProcess this request? (Y/N): ", required=True)
        if confirm and confirm.upper() == 'Y':
            # Remove from queue
            processed_request = self.request_queue.get_next_request()

            # Log to file
            self.log_processed_request(processed_request)

            console.print("\n[green]Request processed successfully![/green]")
            console.print(f"[dim]Remaining requests: {self.request_queue.size()}[/dim]")

            # Add to processed requests (for undo functionality)
            self.request_queue.processed_requests.append(processed_request)

            action_logger.info(
                f"Processed request for employee {processed_request.employee_id}: "
                f"{processed_request.request_type} (Priority: {processed_request.priority_level})"
            )
        else:
            console.print("[yellow]Processing cancelled.[/yellow]")

    def log_processed_request(self, request):
        """Log processed request to a file"""
        log_entry = {
            'employee_id': request.employee_id,
            'request_type': request.request_type,
            'priority': request.priority_level,
            'details': request.request_details,
            'timestamp': request.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)

        # Append to processed requests log
        log_file = 'logs/processed_requests.log'
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            system_logger.error(f"Failed to log processed request: {str(e)}")
            console.print("[red]Error logging request.[/red]")

    def undo_last_action(self):
        """Undo last queue action"""
        if self.request_history.undo(self.request_queue):
            console.print("[green]Undo successful![/green]")
        else:
            console.print("[yellow]Nothing to undo.[/yellow]")

    def redo_last_action(self):
        """Redo last undone action"""
        if self.request_history.redo(self.request_queue):
            console.print("[green]Redo successful![/green]")
        else:
            console.print("[yellow]Nothing to redo.[/yellow]")

    def show_department_hierarchy(self):
        """Display the organizational department tree"""
        console.print("\n[bold blue]Department Hierarchy[/bold blue]")
        self.department_tree.print_tree()

        # Show department selection option
        dept_name = self.get_input("\nEnter department name to view details (or leave blank): ")
        if dept_name:
            dept_node = self.department_tree.find_department(dept_name)
            if dept_node:
                console.print(f"\n[bold]{dept_node.name} Department[/bold]")
                console.print(f"Employees: {len(dept_node.employees)}")

                if dept_node.employees:
                    table = Table(show_header=True, header_style="bold blue")
                    table.add_column("ID")
                    table.add_column("Name")
                    table.add_column("Programmes")

                    for emp in dept_node.employees:
                        table.add_row(
                            str(emp.employee_id),
                            emp.name,
                            ", ".join(emp.enrolled_programmes) if emp.enrolled_programmes else "None"
                        )

                    console.print(table)
            else:
                console.print("[red]Department not found.[/red]")

    def validate_email(self, email):
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email.lower()):
            return False
        return not any(emp.email == email.lower() for emp in self.employees.values())

    def validate_employee_id(self, employee_id):
        return employee_id not in self.employees and len(str(employee_id)) == 5

    def add_employee(self):
        console.print("\n[bold blue]Add New Employee[/bold blue]")
        console.print("[italic]Enter 'back' or '0' at any time to cancel[/italic]\n")

        # Name
        name = self.get_input("Enter employee name: ", required=True)
        if name is None:
            return

        # Employee ID
        while True:
            employee_id = self.get_input("Enter 5-digit employee ID: ", required=True)
            if employee_id is None:
                return

            try:
                employee_id = int(employee_id)
                if self.validate_employee_id(employee_id):
                    break
                console.print("[red]Invalid ID. Either exists or not 5 digits.[/red]")
            except ValueError:
                console.print("[red]Please enter a valid integer.[/red]")

        # Email
        while True:
            email = self.get_input("Enter employee email: ", required=True)
            if email is None:
                return
            if self.validate_email(email):
                break
            console.print("[red]Invalid email format or email already exists.[/red]")

        # Department
        department = self.get_input("Enter department: ", required=True)
        if department is None:
            return
        department = department.title()

        # Status
        while True:
            status = self.get_input("Full-time? (Y/N): ", required=True)
            if status is None:
                return  # User chose to cancel

            status = status.upper().strip()
            if status == 'Y':
                full_time = True
                break
            elif status == 'N':
                full_time = False
                break
            else:
                console.print("[red]Please enter only Y or N.[/red]")
                continue

        # Create and save employee
        new_employee = Employee(name, employee_id, email, department, full_time)
        self.employees[employee_id] = new_employee

        # Ask if user wants to add another employee
        while True:
            choice = self.get_input("\nAdd another employee? (Y/N): ")
            if choice and choice.upper() == 'Y':
                self.save_data()  # Save before adding another
                self.add_employee()
                return
            elif choice and choice.upper() == 'N':
                self.save_data()  # Save before exiting
                console.print(f"\n[green]Employee {name} added successfully![/green]")
                return
            else:
                console.print("[red]Please enter Y or N.[/red]")

    def get_input(self, prompt, required=False, password=False, allow_back=True, input_type=None,
                  min_value=None, max_value=None, max_length=None):
        """Get validated user input with comprehensive validation options"""
        while True:
            try:
                if password:
                    user_input = getpass.getpass(prompt)
                else:
                    print(prompt, end='', flush=True)
                    user_input = input().strip()
                # ... rest of the method remains the same ...

                # Check for back command
                if allow_back and (user_input.lower() == 'back' or user_input == '0'):
                    return None

                # Check required field
                if required and not user_input:
                    console.print("[red]This field is required.[/red]")
                    continue

                # Validate based on input type
                if input_type is not None:
                    if input_type == int:
                        try:
                            user_input = int(user_input)
                            if min_value is not None and user_input < min_value:
                                console.print(f"[red]Value must be at least {min_value}.[/red]")
                                continue
                            if max_value is not None and user_input > max_value:
                                console.print(f"[red]Value must be no more than {max_value}.[/red]")
                                continue
                        except ValueError:
                            console.print("[red]Please enter a valid integer.[/red]")
                            continue

                    elif input_type == float:
                        try:
                            user_input = float(user_input)
                            if min_value is not None and user_input < min_value:
                                console.print(f"[red]Value must be at least {min_value}.[/red]")
                                continue
                            if max_value is not None and user_input > max_value:
                                console.print(f"[red]Value must be no more than {max_value}.[/red]")
                                continue
                        except ValueError:
                            console.print("[red]Please enter a valid number.[/red]")
                            continue

                    elif input_type == 'email':
                        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', user_input):
                            console.print("[red]Please enter a valid email address.[/red]")
                            continue

                    elif input_type == 'name':
                        if not re.match(r'^[a-zA-Z\s\-\.\']+$', user_input):
                            console.print(
                                "[red]Names can only contain letters, spaces, hyphens, and apostrophes.[/red]")
                            continue
                        if max_length and len(user_input) > max_length:
                            console.print(f"[red]Name must be no more than {max_length} characters.[/red]")
                            continue

                # General length validation
                if max_length is not None and len(user_input) > max_length:
                    console.print(f"[red]Input must be no more than {max_length} characters.[/red]")
                    continue

                return user_input

            except EOFError:
                return None
            except KeyboardInterrupt:
                console.print("\n[red]Operation cancelled.[/red]")
                return None

    def get_display_email(self, employee):
        """Safe email decryption with error handling"""
        if not employee.email:
            return "N/A"
        try:
            return self.decrypt_sensitive_data(employee.email)
        # In decrypt_sensitive_data() method, add:
            system_logger.debug(f"Decrypting: {encrypted_text[:20]}...")
            system_logger.debug(f"Using key: {self.encryption_key[:10]}...")
        except Exception as e:
            system_logger.error(f"Email decryption failed for {employee.employee_id}: {str(e)}")
            return "[DECRYPTION ERROR]"

    def get_decrypted_email(self, employee):
        try:
            return self.decrypt_sensitive_data(employee.email)
        except Exception as e:
            system_logger.error(f"Failed to decrypt email: {str(e)}")
            return "[DECRYPTION ERROR]"

    def display_employee_table(self, employees, extra_columns=None):
        """Display employee data in table format with proper email handling"""
        table_data = []
        for emp in employees:
            # Decrypt the email before display
            decrypted_email = self.decrypt_sensitive_data(emp.email) if emp.email else "N/A"

            # Get badge information
            try:
                with shelve.open('badges_db', flag='r') as badges_db:
                    badges = badges_db.get(str(emp.employee_id), [])
                    badge_count = len(badges)
                    points = self.calculate_points(badges)
                    badge_names = ", ".join(badges) if badges else "None"
            except:
                badge_count = 0
                points = 0
                badge_names = "None"

            row = [
                emp.employee_id,
                emp.name,
                decrypted_email,  # Use the decrypted email here
                "Full-time" if emp.full_time_status else "Part-time",
                emp.department,
                len(emp.enrolled_programmes),
                ", ".join(emp.enrolled_programmes) if emp.enrolled_programmes else "None",
                badge_count,
                points,
                badge_names
            ]

            if extra_columns:
                for col in extra_columns:
                    row.append(col(emp))

            table_data.append(row)

        # Rest of the method remains the same...
        headers = [
            "ID", "Name", "Email", "Status", "Department",
            "# Programmes", "Programmes", "Badges", "Points", "Badge Names"
        ]

        if extra_columns:
            headers.extend([col.__name__ for col in extra_columns])

        # Print using tabulate with logging
        action_logger.info("Displayed employee records")
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

    def display_all_employees(self):
        """Display all employee information including ID, name, email, status, programs, and badges"""
        console.print("\n[bold blue]All Employees[/bold blue]")

        if not self.employees:
            console.print("[yellow]No employees found.[/yellow]")
            action_logger.info("Displayed empty employee list")
            return

        self.display_employee_table(sorted(self.employees.values(), key=lambda x: x.employee_id))

    def enrol_programme(self):
        console.print("[italic]Enter 'back' or '0' at any time to cancel[/italic]\n")
        employee_id = self.select_employee()
        if not employee_id:
            return

        employee = self.employees[employee_id]

        while True:
            console.print("\n[bold]Current Enrollments:[/bold]")
            if employee.enrolled_programmes:
                for i, prog in enumerate(employee.enrolled_programmes, 1):
                    console.print(f"{i}. {prog}")
            else:
                console.print("  No current enrollments")

            console.print("\n[bold]Options:[/bold]")
            console.print("1. Add new program")
            console.print("2. Get AI recommendations")
            console.print("3. Done enrolling")

            choice = self.get_input("Select option (1-3): ")
            if choice == '3' or choice is None:
                break

            if choice == '1':
                programme = self.get_input("Enter programme name to enroll: ", required=True)
                if not programme:
                    continue

                if employee.add_training_programme(programme):
                    self.save_data()
                    console.print(f"\n[green]Successfully enrolled in {programme}[/green]")
                    console.print("\n[bold]Updated Enrollment Status:[/bold]")
                    employee.display_details()

            elif choice == '2':
                # Get AI recommendations
                console.print("\n[bold]Getting AI recommendations...[/bold]")
                recommendations = self.program_advisor.get_recommendations(employee)

                if recommendations:
                    console.print("\n[bold]Recommended Programs:[/bold]")
                    for i, rec in enumerate(recommendations, 1):
                        console.print(f"{i}. {rec}")

                    # Allow selecting a recommendation
                    rec_choice = self.get_input("\nSelect recommendation to enroll (number) or enter to skip: ")
                    if rec_choice:
                        try:
                            rec_idx = int(rec_choice) - 1
                            if 0 <= rec_idx < len(recommendations):
                                programme = recommendations[rec_idx]
                                if employee.add_training_programme(programme):
                                    self.save_data()
                                    console.print(f"\n[green]Successfully enrolled in {programme}[/green]")
                                    console.print("\n[bold]Updated Enrollment Status:[/bold]")
                                    employee.display_details()
                        except ValueError:
                            console.print("[red]Please enter a valid number.[/red]")
                else:
                    console.print("\n[yellow]No recommendations available.[/yellow]")

    def unenroll_programme(self):
        console.print("[italic]Enter 'back' or '0' at any time to cancel[/italic]\n")
        employee_id = self.select_employee()
        if not employee_id:
            return

        employee = self.employees[employee_id]
        if not employee.enrolled_programmes:
            console.print("[yellow]Employee is not enrolled in any programmes.[/yellow]")
            return

        while True:
            console.print("\n[bold]Current Enrollments:[/bold]")
            for i, prog in enumerate(employee.enrolled_programmes, 1):
                console.print(f"{i}. {prog}")

            console.print("\n[bold]Options:[/bold]")
            console.print("1. Unenroll from a program")
            console.print("2. Done unenrolling")

            choice = self.get_input("Select option (1-2): ")
            if choice == '2' or choice is None:
                break

            if choice == '1':
                prog_choice = self.get_input("Select programme to unenroll (number): ", required=True)
                if prog_choice is None:
                    continue

                try:
                    prog_choice = int(prog_choice) - 1
                    if 0 <= prog_choice < len(employee.enrolled_programmes):
                        programme = employee.enrolled_programmes[prog_choice]
                        if employee.remove_training_programme(programme):
                            self.save_data()
                            console.print(f"\n[green]Successfully unenrolled from {programme}[/green]")
                            # Show updated status
                            console.print("\n[bold]Updated Enrollment Status:[/bold]")
                            employee.display_details()
                        else:
                            console.print("\n[red]Failed to unenroll from programme.[/red]")
                    else:
                        console.print("[red]Invalid selection.[/red]")
                except ValueError:
                    console.print("[red]Please enter a valid number.[/red]")

    def select_employee(self):
        while True:
            employee_id = self.get_input("Enter employee ID: ", required=True)
            if employee_id is None:
                return None
            try:
                employee_id = int(employee_id)
                if employee_id in self.employees:
                    return employee_id
                console.print("[red]Employee ID not found.[/red]")
            except ValueError:
                console.print("[red]Please enter a valid ID.[/red]")

    def modify_employee(self):
        """Modify an employee's information"""
        console.print("[italic]Enter 'back' or '0' at any time to cancel[/italic]\n")
        employee_id = self.select_employee()
        if not employee_id:
            return

        employee = self.employees[employee_id]
        self.display_employee_table([employee])
        action_logger.info(f"Displayed employee {employee_id} for modification")

        console.print("\n[bold]What would you like to modify?[/bold]")
        console.print("1. Name\n2. Employee ID\n3. Email\n4. Department\n5. Status")
        choice = self.get_input("Enter choice (1-6): ")
        if choice is None or choice == '6':
            action_logger.info("Modification cancelled")
            return

        if choice == '1':
            new_name = self.get_input(f"Current name: {employee.name}\nNew name: ", required=True)
            if new_name:
                old_name = employee.name
                employee.name = new_name
                self.save_data()
                action_logger.info(f"Modified employee {employee_id}: Changed name from {old_name} to {new_name}")
                console.print("[green]Name updated successfully![/green]")
                self.display_employee_table([employee])

        elif choice == '2':
            while True:
                new_id = self.get_input(f"Current ID: {employee.employee_id}\nNew ID: ", required=True)
                if new_id is None:
                    break
                try:
                    new_id = int(new_id)
                    if new_id == employee.employee_id:
                        console.print("[yellow]Same as current ID. No change made.[/yellow]")
                        break
                    if self.validate_employee_id(new_id):
                        # Remove old employee and add with new ID
                        del self.employees[employee.employee_id]
                        employee.employee_id = new_id
                        self.employees[new_id] = employee
                        self.save_data()
                        action_logger.info(f"Changed employee ID from {employee_id} to {new_id}")
                        console.print("[green]Employee ID updated successfully![/green]")
                        break
                    console.print("[red]Invalid ID. Either exists or not 5 digits.[/red]")
                except ValueError:
                    console.print("[red]Please enter a valid integer.[/red]")

        elif choice == '3':
            while True:
                new_email = self.get_input(f"Current email: {employee.email}\nNew email: ", required=True)
                if new_email is None:
                    break
                if self.validate_email(new_email) or new_email.lower() == employee.email:
                    old_email = employee.email
                    employee.email = new_email.lower()
                    self.save_data()
                    action_logger.info(
                        f"Modified employee {employee_id}: Changed email from {old_email} to {new_email}")
                    console.print("[green]Email updated successfully![/green]")
                    break
                console.print("[red]Invalid email format or email already exists.[/red]")

        elif choice == '4':
            new_dept = self.get_input(f"Current department: {employee.department}\nNew department: ", required=True)
            if new_dept:
                old_dept = employee.department
                employee.department = new_dept.title()
                self.save_data()
                action_logger.info(
                    f"Modified employee {employee_id}: Changed department from {old_dept} to {new_dept}"
                )
                console.print("[green]Department updated successfully![/green]")

        elif choice == '5':
            current_status = "Full-time" if employee.full_time_status else "Part-time"
            while True:
                new_status = self.get_input(
                    f"Current status: {current_status}\nNew status (F for Full-time, P for Part-time): ",
                    required=True
                )
                if new_status is None:
                    break
                new_status = new_status.upper()
                if new_status in ['F', 'P']:
                    old_status = employee.full_time_status
                    employee.full_time_status = new_status == 'F'
                    self.save_data()
                    action_logger.info(
                        f"Modified employee {employee_id}: Changed status from {old_status} to {employee.full_time_status}"
                    )
                    console.print("[green]Status updated successfully![/green]")
                    break
                console.print("[red]Please enter F or P.[/red]")

        else:
            console.print("[red]Invalid choice.[/red]")

    def delete_employee(self):
        """Delete an employee from the system"""
        console.print("\n[bold blue]Delete Employee[/bold blue]")
        employee_id = self.select_employee()
        if not employee_id:
            return

        employee = self.employees[employee_id]
        employee.display_details()

        confirm = self.get_input(
            f"\n[red]WARNING:[/red] Are you sure you want to delete {employee.name}? (yes/no): ",
            required=True
        )
        if confirm and confirm.lower() == 'yes':
            del self.employees[employee_id]
            self.save_data()

            # Also remove from badges database if it exists
            try:
                with shelve.open('badges_db') as badges_db:
                    if str(employee_id) in badges_db:
                        del badges_db[str(employee_id)]
            except Exception as e:
                system_logger.error(f"Error deleting employee badges: {str(e)}")

            action_logger.warning(f"Deleted employee: {employee.name} (ID: {employee_id})")
            console.print(f"[green]Employee {employee.name} deleted successfully![/green]")
        else:
            console.print("[yellow]Deletion cancelled.[/yellow]")

    def view_employee_enrollment_history(self):
        """View an employee's enrollment history in table format"""
        console.print("\n[bold blue]View Employee Enrollment History[/bold blue]")

        employee_id = self.select_employee()
        if not employee_id:
            return

        employee = self.employees[employee_id]
        self.display_employee_table([employee])

        # Check if there's any history (either head exists or count > 0)
        if not employee.enrollment_history.head and employee.enrollment_history.count == 0:
            console.print("\n[yellow]No enrollment history found for this employee.[/yellow]")
            return

        # Create enrollment history table
        table = Table(
            title=f"Enrollment History - {employee.name}",
            show_header=True,
            header_style="bold magenta",
            show_lines=True,
            expand=True
        )

        # Add columns to the table
        table.add_column("Programme", width=25)
        table.add_column("Status", width=12)
        table.add_column("Enrollment Date", width=20)
        table.add_column("Completion Date", width=20)
        table.add_column("Feedback", width=10)
        table.add_column("Rating", width=8)

        # Populate the table with enrollment data
        current = employee.enrollment_history.head

        while current:
            # Determine status color
            status_color = "green" if current.status == "Completed" else "red" if current.status == "Dropped" else "yellow"

            # Format dates
            enroll_date = current.enrollment_date.strftime("%Y-%m-%d")
            complete_date = current.completion_date.strftime("%Y-%m-%d") if current.completion_date else "N/A"

            # Format feedback and rating
            feedback = "Yes" if current.feedback else "No"
            rating = str(current.rating) if current.rating else "N/A"

            table.add_row(
                current.programme_name,
                f"[{status_color}]{current.status}[/{status_color}]",
                enroll_date,
                complete_date,
                feedback,
                rating
            )

            current = current.next

        # Display the table
        console.print("\n[bold]Enrollment History:[/bold]")
        console.print(table)

    def bubble_sort_by_department(self):
        """Display employees sorted by department with full information"""
        employee_list = list(self.employees.values())
        n = len(employee_list)

        # Bubble sort implementation with logging
        action_logger.info("Starting bubble sort by department")
        for i in range(n):
            for j in range(0, n - i - 1):
                # Sort by department, then by employee ID
                if (employee_list[j].department > employee_list[j + 1].department or
                        (employee_list[j].department == employee_list[j + 1].department and
                         employee_list[j].employee_id > employee_list[j + 1].employee_id)):
                    employee_list[j], employee_list[j + 1] = employee_list[j + 1], employee_list[j]

        console.print("\n[bold blue]Employees Sorted by Department (A-Z)[/bold blue]")
        self.display_employee_table(employee_list)
        action_logger.info(f"Completed bubble sort for {len(employee_list)} employees")

    def sort_by_status(self):
        """Display employees sorted by employment status (full-time first) using insertion sort"""
        employee_list = list(self.employees.values())

        # Insertion sort implementation with logging
        action_logger.info("Starting insertion sort by status")
        for i in range(1, len(employee_list)):
            key = employee_list[i]
            j = i - 1
            # Sort by status (full-time first) and then by employee ID
            while j >= 0 and (
                    (employee_list[j].full_time_status < key.full_time_status) or
                    (employee_list[j].full_time_status == key.full_time_status and employee_list[
                        j].employee_id > key.employee_id)
            ):
                employee_list[j + 1] = employee_list[j]
                j -= 1
            employee_list[j + 1] = key

        console.print("\n[bold blue]Employees Sorted by Employment Status[/bold blue]")
        self.display_employee_table(employee_list)
        action_logger.info(f"Completed status sort for {len(employee_list)} employees")

    def selection_sort_by_programmes(self):
        """Display employees sorted by number of programmes with full information"""
        employee_list = list(self.employees.values())
        n = len(employee_list)

        # Selection sort implementation with logging
        action_logger.info("Starting selection sort by programmes")
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                # Sort by number of programmes (descending), then by employee ID (ascending)
                if (len(employee_list[j].enrolled_programmes) > len(employee_list[min_idx].enrolled_programmes) or
                        (len(employee_list[j].enrolled_programmes) == len(
                            employee_list[min_idx].enrolled_programmes) and
                         employee_list[j].employee_id < employee_list[min_idx].employee_id)):
                    min_idx = j
            employee_list[i], employee_list[min_idx] = employee_list[min_idx], employee_list[i]

        console.print("\n[bold blue]Employees Sorted by Number of Programmes[/bold blue]")
        self.display_employee_table(employee_list)
        action_logger.info(f"Completed selection sort for {len(employee_list)} employees")

    def search_employee(self):
        console.print("\n[bold blue]Search Employee[/bold blue]")
        console.print("1. By ID\n2. By Name")
        choice = self.get_input("Choose search method (1-2): ")
        if choice is None:
            return

        if choice == '1':
            employee_id = self.select_employee()
            if employee_id:
                action_logger.info(f"Searched for employee by ID: {employee_id}")
                self.display_employee_table([self.employees[employee_id]])
        elif choice == '2':
            name = self.get_input("Enter name (or part): ")
            if name is None:
                return
            name = name.lower()
            found = []
            for emp in self.employees.values():
                if name in emp.name.lower():
                    found.append(emp)

            if found:
                action_logger.info(f"Searched for employees by name: '{name}' - found {len(found)} matches")
                self.display_employee_table(found)
            else:
                console.print("[yellow]No matching employees found.[/yellow]")
                action_logger.info(f"Searched for employees by name: '{name}' - no matches")
        else:
            console.print("[red]Invalid choice.[/red]")
            action_logger.warning("Invalid search choice selected")

    def filter_by_programme(self):
        """Filter employees by training programme"""
        console.print("\n[bold blue]Filter by Training Programme[/bold blue]")
        programme = self.get_input("Enter programme name to filter: ", required=True)
        if not programme:
            return

        programme_lower = programme.lower()
        filtered = []
        for emp in self.employees.values():
            if any(p.lower() == programme_lower for p in emp.enrolled_programmes):
                filtered.append(emp)

        if filtered:
            # Sort filtered results by employee ID
            filtered.sort(key=lambda x: x.employee_id)
            console.print(f"\n[bold]Employees enrolled in {programme}:[/bold]")
            self.display_employee_table(filtered)
            action_logger.info(f"Filtered employees by programme: {programme}")
        else:
            console.print(f"[yellow]No employees enrolled in {programme}.[/yellow]")
            action_logger.info(f"No employees found for programme filter: {programme}")

    def login(self):
        console.print("\n[bold blue]Login[/bold blue]")
        console.print("1. Admin\n2. Employee\n3. Biometric Login\n4. Exit")

        while True:
            choice = self.get_input("Enter choice (1-4): ", required=True)
            if choice is None:
                continue

            elif choice == '1':
                # Admin login
                username = self.get_input("Username: ", required=True)
                if username is None:
                    continue

                password = self.get_input("Password: ", required=True)  # Removed password=True
                if password is None:
                    continue

                if username == self.admin_username and password == self.admin_password:
                    self.logged_in_user = "admin"
                    action_logger.info("Admin logged in")
                    console.print("\n[green]Admin login successful![/green]")
                    return True
                else:
                    console.print("[red]Invalid credentials.[/red]")
                    return False

            elif choice == '2':
                employee_id = self.select_employee()
                if not employee_id:
                    console.print("[red]Invalid Employee ID.[/red]")
                    return False

                employee = self.employees[employee_id]

                if employee.password is None:
                    console.print("\n[yellow]First-time login. Set your password.[/yellow]")
                    while True:
                        password = self.get_input("Set password: ", required=True)  # Removed password=True
                        if password is None:
                            break

                        confirm = self.get_input("Confirm password: ", required=True)  # Removed password=True
                        if confirm is None:
                            break

                        if password == confirm:
                            # Hash the password with bcrypt
                            hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                            employee.password = hashed.decode('utf-8')
                            self.save_data()
                            self.logged_in_user = employee_id
                            action_logger.info(f"Employee {employee_id} set password")
                            console.print("\n[green]Password set and login successful![/green]")
                            return True
                        else:
                            console.print("[red]Passwords do not match. Try again.[/red]")
                else:
                    password = self.get_input("Enter password: ", required=True)  # Removed password=True
                    if password is None:
                        continue

                    # Verify the password against the stored hash
                    if bcrypt.checkpw(password.encode('utf-8'), employee.password.encode('utf-8')):
                        self.logged_in_user = employee_id
                        action_logger.info(f"Employee {employee_id} logged in")
                        console.print("\n[green]Login successful![/green]")
                        return True
                    else:
                        console.print("[red]Incorrect password.[/red]")
                        return False

            elif choice == '3':
                if self.biometric_login():
                    return True
                else:
                    continue

            elif choice == '4':
                self.save_data()
                console.print("\n[blue]Goodbye![/blue]")
                sys.exit(0)
            else:
                console.print("[red]Invalid choice.[/red]")
                continue

    def show_employee_details(self, employee):
        """Display all details for the logged-in employee in list format"""
        console.print(f"\n[bold blue]Employee Details - {employee.name}[/bold blue]")
        console.print("─" * 50)  # Divider line

        # Get badge information
        try:
            with shelve.open('badges_db', flag='r') as badges_db:
                badges = badges_db.get(str(employee.employee_id), [])
                badge_count = len(badges)
                points = self.calculate_points(badges)
                badge_names = ", ".join(badges) if badges else "None"
        except Exception as e:
            console.print(f"[red]Error accessing badges: {str(e)}[/red]")
            badge_count = 0
            points = 0
            badge_names = "None"

        # Display details in list format
        console.print(f"[bold]Employee ID:[/bold] {employee.employee_id}")
        console.print(f"[bold]Name:[/bold] {employee.name}")
        console.print(f"[bold]Email:[/bold] {employee.email}")
        console.print(f"[bold]Department:[/bold] {employee.department}")
        console.print(
            f"[bold]Status:[/bold] {'[green]Full-time[/green]' if employee.full_time_status else '[yellow]Part-time[/yellow]'}")

        # Enrolled Programmes
        console.print("\n[bold]Enrolled Programmes:[/bold]")
        if employee.enrolled_programmes:
            for i, programme in enumerate(employee.enrolled_programmes, 1):
                console.print(f"  {i}. {programme}")
        else:
            console.print("  No programmes enrolled")

        # Badges
        console.print(f"\n[bold]Badges:[/bold] {badge_count} (Total Points: [green]{points}[/green])")
        if badge_count > 0:
            console.print("  " + badge_names)
        else:
            console.print("  No badges earned")

        # Pending Requests
        console.print("\n[bold]Pending Requests:[/bold]")
        if employee.pending_requests:
            for req in employee.pending_requests:
                status_color = "yellow" if req['status'] == 'Pending' else "green" if req[
                                                                                          'status'] == 'Approved' else "red"
                console.print(
                    f"  • {req['programme']} ([{status_color}]{req['status']}[/{status_color}]) - {req['date']}")
        else:
            console.print("  No pending requests")

        console.print("─" * 50)  # Divider line

    def submit_feedback(self, employee):
        if not employee.enrolled_programmes:
            console.print("[yellow]No programmes to provide feedback.[/yellow]")
            return

        console.print("\n[bold blue]Submit Feedback[/bold blue]")
        for i, prog in enumerate(employee.enrolled_programmes, 1):
            console.print(f"{i}. {prog}")

        prog_choice = self.get_input("Select programme: ", required=True)
        if prog_choice is None:
            return

        try:
            prog_choice = int(prog_choice) - 1
            if 0 <= prog_choice < len(employee.enrolled_programmes):
                programme = employee.enrolled_programmes[prog_choice]
                feedback = self.get_input("Your feedback: ", required=True)
                if feedback is None:
                    return

                while True:
                    rating = self.get_input("Rating (1-5): ", required=True)
                    if rating is None:
                        return
                    try:
                        rating = int(rating)
                        if 1 <= rating <= 5:
                            break
                        console.print("[red]Rating must be 1-5.[/red]")
                    except ValueError:
                        console.print("[red]Enter a number.[/red]")

                employee.add_feedback(programme, feedback, rating)
                self.save_data()
                console.print("[green]Thank you for your feedback![/green]")
            else:
                console.print("[red]Invalid selection.[/red]")
        except ValueError:
            console.print("[red]Enter a valid number.[/red]")

    def request_enrollment(self, employee):
        """Enhanced request enrollment with AI recommendations"""
        console.print("\n[bold blue]Request Program Enrollment[/bold blue]")
        console.print("1. Request specific program\n2. Get AI recommendations\n3. Back")

        choice = self.get_input("Select option: ")
        if choice == '1':
            programme = self.get_input("Enter program name: ", required=True)
            if programme:
                if employee.request_enrollment(self, programme):
                    self.save_data()
                    console.print(f"\n[green]Request submitted for {programme}.[/green]")
                else:
                    console.print(f"\n[yellow]Already enrolled or pending for {programme}.[/yellow]")

        elif choice == '2':
            # Get AI recommendations and handle the request
            if employee.request_enrollment(self):
                self.save_data()
                console.print("\n[green]Request submitted successfully![/green]")

        elif choice == '3':
            return

    def change_password(self, employee):
        current = self.get_input("Current password: ", required=True)
        if current is None:
            return

        if current != employee.password:
            console.print("[red]Incorrect password.[/red]")
            return

        new_pass = self.get_input("New password: ", required=True)
        if new_pass is None:
            return

        confirm = self.get_input("Confirm password: ", required=True)
        if confirm is None:
            return

        if new_pass == confirm:
            employee.password = new_pass
            self.save_data()
            action_logger.info(f"Employee {employee.employee_id} changed password")
            console.print("[green]Password changed![/green]")
        else:
            console.print("[red]Passwords don't match.[/red]")

    def view_employee_badges(self, employee):
        """Display badges for the logged-in employee"""
        try:
            with shelve.open('badges_db', flag='r') as badges_db:
                badges = badges_db.get(str(employee.employee_id), [])

                console.print("\n[bold blue]Your Badges[/bold blue]")
                if not badges:
                    console.print("[yellow]No badges earned yet.[/yellow]")
                    return

                table = Table(show_header=True, header_style="bold blue", show_lines=True)
                table.add_column("Badge", style="cyan")
                table.add_column("Points", style="green")

                # Define badge values for point calculation
                badge_values = {
                    'Fast Learner': 20,
                    'Innovation Learner': 15,
                    'Most Passionate': 10,
                    'Most Initiative': 25,
                    # Default value for unknown badges
                    'default': 10
                }

                for badge in badges:
                    # Calculate points for each badge
                    points = badge_values.get(badge, badge_values['default'])
                    table.add_row(badge, str(points))

                console.print(table)
                console.print(f"\n[green]Total Points: {self.calculate_points(badges)}[/green]")

        except Exception as e:
            console.print(f"[red]Error viewing badges: {str(e)}[/red]")
            system_logger.error(f"Error viewing badges for {employee.employee_id}: {str(e)}")

    def view_feedback(self):
        """Display feedback in a properly formatted table with solid lines and consistent columns"""
        console.print("\n[bold blue]Programme Feedback[/bold blue]")

        # Collect all feedback data
        feedback_data = []
        for emp in self.employees.values():
            if emp.feedback:
                for fb in emp.feedback:
                    feedback_data.append({
                        'employee_id': emp.employee_id,
                        'name': emp.name,
                        'programme': fb['programme'],
                        'rating': fb['rating'],
                        'feedback': fb['feedback'],
                        'date': fb['date']
                    })

        if not feedback_data:
            console.print("[yellow]No feedback available.[/yellow]")
            return

        # Sort feedback by employee ID (ascending)
        feedback_data.sort(key=lambda x: x['employee_id'])

        # Prepare table data with proper formatting
        table_data = []
        for fb in feedback_data:
            # Convert rating to stars
            rating = int(fb['rating'])
            stars = "★" * rating + "☆" * (5 - rating)

            # Clean up feedback text
            feedback_text = fb['feedback'].replace('\n', ' ').strip()
            if len(feedback_text) > 50:
                feedback_text = feedback_text[:47] + "..."

            # Format date
            try:
                formatted_date = datetime.strptime(fb['date'], '%Y-%m-%d %H:%M:%S').strftime('%d %b %Y %H:%M')
            except:
                formatted_date = fb['date']  # Fallback if parsing fails

            table_data.append([
                fb['employee_id'],
                fb['name'][:20],  # Limit name length
                fb['programme'][:20],  # Limit programme length
                stars,
                feedback_text,
                formatted_date
            ])

        # Define headers
        headers = [
            "ID",
            "Name",
            "Programme",
            "Rating",
            "Feedback",
            "Date"
        ]

        # Print using pretty format with adjusted column widths
        action_logger.info("Displayed feedback records")

        # Create custom column alignment
        colalign = ("left", "left", "left", "left", "left", "left")

        console.print(tabulate(table_data, headers=headers, tablefmt="fancy_grid", colalign=colalign))

    def export_data(self):
        """Export employee data with badges information"""
        console.print("\n[bold blue]Export Employee Data[/bold blue]")

        try:
            # Prepare data and sort by employee ID
            table_data = []

            # Check if badges_db exists
            if not os.path.exists('badges_db.dat'):
                # Create empty database
                with shelve.open('badges_db', flag='c') as db:
                    pass

            with shelve.open('badges_db', flag='r') as badges_db:
                # Rest of your existing code...
                # Create list of employees sorted by ID
                sorted_employees = sorted(self.employees.values(), key=lambda emp: emp.employee_id)

                for emp in sorted_employees:
                    badges = badges_db.get(str(emp.employee_id), [])

                    table_data.append([
                        emp.employee_id,
                        emp.name,
                        emp.department,
                        "Full-time" if emp.full_time_status else "Part-time",
                        emp.email,
                        ", ".join(emp.enrolled_programmes) if emp.enrolled_programmes else "None",
                        len(badges),
                        self.calculate_points(badges),
                        ", ".join(badges) if badges else "None"
                    ])

            if not table_data:
                console.print("[yellow]No employee data to export.[/yellow]")
                return

            # Display preview
            headers = ["ID", "Name", "Department", "Status", "Email", "Programmes", "Badges", "Points", "Badge Names"]

            preview_table = Table(show_header=True, header_style="bold blue", show_lines=True, expand=True)
            for header in headers:
                preview_table.add_column(header)

            for row in table_data[:5]:  # Show first 5 as preview
                preview_table.add_row(*[str(item) for item in row])

            console.print(preview_table)

            if len(table_data) > 5:
                console.print(f"[dim]Showing 5 of {len(table_data)} records...[/dim]")

            # Export options
            console.print("\n1. Export to Excel\n2. Export to PDF\n3. Back")
            choice = self.get_input("Select export format: ")

            if choice == '1':
                self.export_to_excel(sorted_employees)  # Pass sorted employees
            elif choice == '2':
                self.export_to_pdf(table_data, headers)

        except Exception as e:
            console.print(f"[red]Error preparing export data: {str(e)}[/red]")
            system_logger.error(f"Export preparation failed: {str(e)}")

    def export_to_excel(self, sorted_employees):
        """Export employee data to Excel with all requested fields"""
        try:
            # Prepare data with all required fields, already sorted
            data = []
            with shelve.open('badges_db', flag='r') as badges_db:
                for emp in sorted_employees:
                    # Get badge information
                    badges = badges_db.get(str(emp.employee_id), [])
                    badge_count = len(badges)
                    points = self.calculate_points(badges)
                    badge_names = ", ".join(badges) if badges else "None"

                    data.append({
                        'ID': emp.employee_id,
                        'Name': emp.name,
                        'Department': emp.department,
                        'Status': 'Full-time' if emp.full_time_status else 'Part-time',
                        'Email': emp.email,
                        'Programmes': ', '.join(emp.enrolled_programmes) if emp.enrolled_programmes else 'None',
                        'Badges': badge_count,
                        'Points': points,
                        'Badge Names': badge_names
                    })

            # Create DataFrame
            df = pd.DataFrame(data)

            # Ensure exports directory exists
            os.makedirs('exports', exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_path = os.path.join('exports', f'employee_export_{timestamp}.xlsx')

            # Export to Excel
            try:
                # Try with openpyxl first
                df.to_excel(export_path, index=False, engine='openpyxl')
            except ImportError:
                # Fallback to default engine
                df.to_excel(export_path, index=False)

            action_logger.info(f"Exported data to Excel: {export_path}")
            console.print(f"[green]Data successfully exported to:[/green] [bold]{export_path}[/bold]")

        except Exception as e:
            console.print(f"[red]Error exporting to Excel: {str(e)}[/red]")
            system_logger.error(f"Excel export failed: {str(e)}")
            if "openpyxl" in str(e):
                console.print("[yellow]Note: Install openpyxl for better Excel support: pip install openpyxl[/yellow]")

    def export_to_pdf(self, table_data, headers):
        """Export data to PDF file"""
        try:
            # Create PDF
            pdf = FPDF(orientation='L')  # Landscape
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()

            # Title
            pdf.set_font("Helvetica", 'B', 16)
            pdf.cell(0, 10, "Employee Training Report", 0, 1, 'C')
            pdf.set_font("Helvetica", '', 10)
            pdf.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
            pdf.ln(10)

            # Calculate column widths
            col_widths = [15, 30, 25, 20, 50, 40, 15, 15, 40]  # Adjusted for ID first

            # Header row
            pdf.set_fill_color(200, 220, 255)
            pdf.set_font("Helvetica", 'B', 10)
            for i, header in enumerate(headers):
                pdf.cell(col_widths[i], 10, header, 1, 0, 'C', True)
            pdf.ln()

            # Data rows (already sorted)
            pdf.set_font("Helvetica", '', 8)
            fill = False
            for row in table_data:
                fill = not fill
                pdf.set_fill_color(240, 240, 240) if fill else pdf.set_fill_color(255, 255, 255)

                for i, item in enumerate(row):
                    pdf.cell(col_widths[i], 10, str(item), 1, 0, 'L', fill)
                pdf.ln()

                if pdf.get_y() > 270:  # New page if near bottom
                    pdf.add_page()
                    # Redraw header
                    pdf.set_fill_color(200, 220, 255)
                    pdf.set_font("Helvetica", 'B', 10)
                    for i, header in enumerate(headers):
                        pdf.cell(col_widths[i], 10, header, 1, 0, 'C', True)
                    pdf.ln()
                    pdf.set_font("Helvetica", '', 8)

            # Save file
            export_path = os.path.join('exports', f'employee_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')
            pdf.output(export_path)

            console.print(f"[green]Successfully exported to {export_path}[/green]")
            action_logger.info(f"Exported employee data to PDF: {export_path}")

        except Exception as e:
            console.print(f"[red]Error exporting to PDF: {str(e)}[/red]")
            system_logger.error(f"PDF export failed: {str(e)}")

    def launch_external_dashboard(self):
        """Launch a web-based dashboard with visualizations"""
        try:
            # Create Flask app with correct template path
            app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

            # Suppress Flask's default logging
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)

            self.dashboard_app = app
            self.stop_dashboard = False
            self.dashboard_data = {}

            @app.route('/')
            def dashboard():
                try:
                    data = self._update_dashboard_data()
                    # Convert data to ensure JSON serialization
                    serializable_data = {
                        'metrics': data,
                        'department_data': [
                            {'department': k, 'count': v}
                            for k, v in data['department_distribution'].items()
                        ],
                        'program_data': [
                            {'program': k, 'enrollments': v}
                            for k, v in data['program_popularity'].items()
                        ],
                        'badge_data': [
                            {'badge': k, 'count': v}
                            for k, v in data['badge_stats']['badge_distribution'].items()
                        ]
                    }
                    return render_template('dashboard.html', **serializable_data)
                except Exception as e:
                    return f"Error generating dashboard: {str(e)}", 500

            @app.route('/api/dashboard')
            def api_dashboard():
                try:
                    data = self._update_dashboard_data()
                    return jsonify(data)
                except Exception as e:
                    return jsonify({"error": str(e)}), 500

            # Start server thread
            def run_server():
                app.run(port=5000, use_reloader=False)

            server_thread = threading.Thread(target=run_server)
            server_thread.daemon = True
            server_thread.start()

            time.sleep(1)
            webbrowser.open('http://localhost:5000')

            console.print("\n[green]Dashboard launched![/green]")

        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            system_logger.error(f"Dashboard error: {str(e)}")

    def _update_dashboard_data(self):
        """Update the dashboard data structure with current metrics"""
        metrics = self.real_time_dashboard.update_metrics()

        # Calculate department breakdown with simple counts (for pie chart)
        dept_counts = {}
        for emp in self.employees.values():
            dept_counts[emp.department] = dept_counts.get(emp.department, 0) + 1

        # Calculate full-time vs part-time breakdown (for potential future use)
        dept_breakdown = defaultdict(lambda: {'full_time': 0, 'part_time': 0})
        for emp in self.employees.values():
            if emp.full_time_status:
                dept_breakdown[emp.department]['full_time'] += 1
            else:
                dept_breakdown[emp.department]['part_time'] += 1

        # Convert all data to basic Python types that are JSON serializable
        dashboard_data = {
            'total_employees': int(metrics.get('total_employees', 0)),
            'full_time_employees': int(metrics.get('full_time_employees', 0)),
            'part_time_employees': int(metrics.get('part_time_employees', 0)),
            'active_enrollments': int(metrics.get('active_enrollments', 0)),
            'completion_rate': float(metrics.get('completion_rates', {}).get('completion_rate', 0)),
            'request_stats': {
                'total_requests': int(metrics.get('request_stats', {}).get('total_requests', 0)),
                'priority_counts': dict(metrics.get('request_stats', {}).get('priority_counts', {})),
                'type_counts': dict(metrics.get('request_stats', {}).get('type_counts', {}))
            },
            'program_popularity': dict(metrics.get('program_popularity', {})),
            'department_distribution': dict(dept_counts),  # Simple counts for pie chart
            'department_breakdown': dict(dept_breakdown),  # Detailed breakdown for other visualizations
            'badge_stats': dict(metrics.get('badge_stats', {})),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        return dashboard_data

    def manage_badges(self):
        """Admin function to add/remove badges for employees"""
        console.print("[italic]Enter 'back' or '0' at any time to cancel[/italic]\n")

        # Define available badge types with points
        badge_types = {
            'Fast Learner': 20,
            'Innovation Learner': 15,
            'Most Passionate': 10,
            'Most Initiative': 25,
        }

        employee_id = self.select_employee()
        if not employee_id:
            return

        employee = self.employees[employee_id]

        try:
            with shelve.open('badges_db') as badges_db:
                badges = badges_db.get(str(employee_id), [])

                while True:
                    console.print(f"\n[bold]Current Badges for {employee.name}:[/bold]")
                    if badges:
                        for i, badge in enumerate(badges, 1):
                            console.print(f"{i}. [cyan]{badge}[/cyan]")
                    else:
                        console.print("[yellow]No badges assigned[/yellow]")

                    # Display available badge types
                    console.print("\n[bold]Available Badge Types:[/bold]")
                    for badge, points in badge_types.items():
                        console.print(f"- {badge}: [yellow]{points}[/yellow] points")

                    console.print("\n1. Add Badge\n2. Remove Badge\n3. Back")
                    choice = self.get_input("Select option: ")

                    if choice == '1':
                        console.print("\n[bold]Available Badges:[/bold]")
                        for i, (badge, points) in enumerate(badge_types.items(), 1):
                            console.print(f"{i}. {badge} ({points} points)")

                        badge_choice = self.get_input("\nSelect badge to add (number or name): ")
                        if badge_choice:
                            try:
                                # Try numeric selection first
                                badge_idx = int(badge_choice) - 1
                                if 0 <= badge_idx < len(badge_types):
                                    badge_name = list(badge_types.keys())[badge_idx]
                                else:
                                    raise ValueError
                            except ValueError:
                                # Fall back to name input
                                badge_name = badge_choice.title()

                            if badge_name in badge_types:
                                if badge_name not in badges:
                                    badges.append(badge_name)
                                    badges_db[str(employee_id)] = badges
                                    console.print(f"\n[green]Added badge: {badge_name}[/green]")
                                    action_logger.info(f"Added badge '{badge_name}' to employee {employee_id}")
                                else:
                                    console.print("[yellow]Employee already has this badge[/yellow]")
                            else:
                                # Allow custom badges with default points
                                confirm = self.get_input(
                                    f"'{badge_name}' is not a standard badge. Add anyway? (Y/N): "
                                )
                                if confirm and confirm.upper() == 'Y':
                                    badges.append(badge_name)
                                    badges_db[str(employee_id)] = badges
                                    console.print(f"\n[green]Added custom badge: {badge_name}[/green]")
                                    action_logger.info(f"Added custom badge '{badge_name}' to employee {employee_id}")

                    elif choice == '2':
                        if not badges:
                            console.print("[yellow]No badges to remove[/yellow]")
                            continue

                        badge_choice = self.get_input("Enter badge number to remove: ")
                        if badge_choice:
                            try:
                                idx = int(badge_choice) - 1
                                if 0 <= idx < len(badges):
                                    removed = badges.pop(idx)
                                    badges_db[str(employee_id)] = badges
                                    console.print(f"\n[red]Removed badge: {removed}[/red]")
                                    action_logger.info(f"Removed badge '{removed}' from employee {employee_id}")
                                else:
                                    console.print("[red]Invalid selection[/red]")
                            except ValueError:
                                console.print("[red]Please enter a valid number[/red]")

                    elif choice == '3':
                        break

        except Exception as e:
            console.print(f"[red]Error managing badges: {str(e)}[/red]")
            system_logger.error(f"Badge management failed: {str(e)}")

    def calculate_points(self, badges):
        """Calculate points based on badges with different values"""
        badge_values = {
            'excellence': 20,
            'innovation': 15,
            'teamwork': 10,
            'leadership': 25
        }

        total = 0
        for badge in badges:
            total += badge_values.get(badge.lower(), 10)  # Default 10 points for unknown badges

        return total

    def display_export_preview(self, data):
        """Display a preview of the exported data in console"""
        console.print("\n[bold blue]Export Preview[/bold blue]")

        if not data:
            console.print("[yellow]No data to display.[/yellow]")
            return

        # Prepare table data
        headers = ["Name", "ID", "Department", "Status", "Email", "Programmes", "Badges", "Points", "Badge Names"]
        table_data = []

        for row in data[:5]:  # Show first 5 rows as preview
            table_data.append([
                row['Name'],
                row['ID'],
                row['Department'],
                row['Status'],
                row['Email'],
                row['Programmes'],
                row['Badges'],
                row['Points'],
                row['Badge Names'][:30] + "..." if len(row['Badge Names']) > 30 else row['Badge Names']
            ])

        # Create and display table
        table = Table(show_header=True, header_style="bold blue", show_lines=True, expand=True)
        for header in headers:
            table.add_column(header)

        for row in table_data:
            table.add_row(*[str(item) for item in row])

        console.print(table)

        if len(data) > 5:
            console.print(f"[dim]Showing 5 of {len(data)} records...[/dim]")

    def view_pending_requests(self):
        console.print("\n[bold blue]Pending Requests[/bold blue]")
        pending_requests = []

        for emp in self.employees.values():
            for req in emp.pending_requests:
                if req['status'] == 'Pending':
                    pending_requests.append({
                        'employee_id': emp.employee_id,
                        'name': emp.name,
                        'programme': req['programme'],
                        'date': req['date']
                    })

        if not pending_requests:
            console.print("[yellow]No pending requests.[/yellow]")
            return

        table = Table(show_header=True, header_style="bold blue", show_lines=True, expand=True)
        table.add_column("Employee ID", style="dim", width=10)
        table.add_column("Name", width=20)
        table.add_column("Programme", width=30)
        table.add_column("Date", width=20)
        table.add_column("Actions", width=20)

        for req in pending_requests:
            table.add_row(
                str(req['employee_id']),
                req['name'],
                req['programme'],
                req['date'],
                "[green]Approve[/green] / [red]Reject[/red]"
            )

        console.print(table)

        while True:
            action = self.get_input("\nEnter 'approve ID' or 'reject ID' (or '0'): ")
            if action is None or action.lower() == '0':
                break

            parts = action.lower().split()
            if len(parts) == 2 and parts[0] in ['approve', 'reject']:
                try:
                    emp_id = int(parts[1])
                    employee = self.employees.get(emp_id)
                    if employee:
                        for req in employee.pending_requests:
                            if req['status'] == 'Pending':
                                if parts[0] == 'approve':
                                    employee.add_training_programme(req['programme'])
                                    req['status'] = 'Approved'
                                    action_logger.info(f"Approved {emp_id}'s request for {req['programme']}")
                                    console.print(f"[green]Approved {employee.name}'s request.[/green]")
                                else:
                                    req['status'] = 'Rejected'
                                    action_logger.info(f"Rejected {emp_id}'s request for {req['programme']}")
                                    console.print(f"[red]Rejected {employee.name}'s request.[/red]")
                                self.save_data()
                                break
                        else:
                            console.print("[yellow]No pending request found.[/yellow]")
                    else:
                        console.print("[red]Employee not found.[/red]")
                except ValueError:
                    console.print("[red]Invalid employee ID.[/red]")
            else:
                console.print("[red]Invalid command.[/red]")

    def manage_requests_menu(self):
        """Submenu for managing employee requests"""
        while True:
            console.print("\n[bold blue]Employee Requests Management[/bold blue]")
            console.print("1. Add Request\n2. View Queue Statistics\n3. Process Next Request")
            console.print("4. View All Requests\n5. Undo Last Action\n6. Redo Last Action\n7. Back")

            choice = self.get_input("Enter choice (1-7): ")
            if choice == '1':
                self.add_employee_request()
            elif choice == '2':
                self.view_request_statistics()
            elif choice == '3':
                self.process_next_request()
            elif choice == '4':
                self.display_requests_table(self.request_queue.requests)
            elif choice == '5':
                self.undo_last_action()
            elif choice == '6':
                self.redo_last_action()
            elif choice == '7':
                break
            else:
                console.print("[red]Invalid choice.[/red]")

    def admin_menu(self):
        while True:
            console.print("\n[bold blue]Admin Menu[/bold blue]")
            console.print("0. Back to Main Menu\n1. View Employees\n2. Add Employee\n3. Modify Employee")
            console.print("4. Delete Employee\n5. Enroll Employee\n6. Unenroll Employee\n7.View Enrollment History")
            console.print("8. Quick Sort by Department/Name\n9. Merge Sort by Programmes/ID")
            console.print("10. Sort by Status\n11. Search Employee\n12. Filter by Programme")
            console.print("13. View Pending Request \n14. Manage Requests\n15. View Feedback\n16. Manage Badges")
            console.print("17. Export Data\n18. View Dashboard\n19. Launch External Dashboard\n20. Logout\n21. Exit")

            choice = self.get_input("Enter choice (1-21): ")
            if choice is None:
                continue
            if choice == '0' or choice.lower() == 'back':
                break
            if choice == '1':
                self.display_all_employees()
            elif choice == '2':
                self.add_employee()
            elif choice == '3':
                self.modify_employee()
            elif choice == '4':
                self.delete_employee()
            elif choice == '5':
                self.enrol_programme()
            elif choice == '6':
                self.unenroll_programme()
            elif choice == '7':
                self.view_employee_enrollment_history()
            elif choice == '8':
                self.display_sorted_by_department()
            elif choice == '9':
                self.display_sorted_by_programmes()
            elif choice == '10':
                self.sort_by_status()
            elif choice == '11':
                self.search_employee()
            elif choice == '12':
                self.filter_by_programme()
            elif choice == '13':
                self.view_pending_requests()
            elif choice == '14':
                self.manage_requests_menu()
            elif choice == '15':
                self.view_feedback()
            elif choice == '16':
                self.manage_badges()
            elif choice == '17':
                self.export_data()
            elif choice == '18':
                self.real_time_dashboard.display_dashboard()
            elif choice == '19':
                self.launch_external_dashboard()
            elif choice == '20':
                action_logger.info("Admin logged out")
                self.logged_in_user = None
                console.print("[green]Logged out.[/green]")
                break
            elif choice == '21':
                self.save_data()
                console.print("\n[blue]Goodbye![/blue]")
                sys.exit(0)

    def employee_menu(self):
        """Menu for regular employees after login"""
        employee = self.employees[self.logged_in_user]

        while True:
            console.print(f"\n[bold blue]Employee Menu - {employee.name}[/bold blue]")
            console.print("1. View Profile\n2. View Enrollment History\n3. Submit Feedback")
            console.print("4. Request New Enrollment\n5. View Badges\n6. Change Password\n7. Logout")

            choice = self.get_input("Enter choice (1-7): ")
            if choice is None:
                continue

            if choice == '1':
                self.show_employee_details(employee)
            elif choice == '2':
                employee.enrollment_history.display_history()
            elif choice == '3':
                self.submit_feedback(employee)
            elif choice == '4':
                self.request_enrollment(employee)
            elif choice == '5':
                self.view_employee_badges(employee)
            elif choice == '6':
                self.change_password(employee)
            elif choice == '7':
                action_logger.info(f"Employee {employee.employee_id} logged out")
                self.logged_in_user = None
                console.print("[green]Logged out successfully![/green]")
                break
            else:
                console.print("[red]Invalid choice.[/red]")


def show_banner():
    banner_text = pyfiglet.figlet_format("Employee Academy", font="slant")
    console.print(banner_text, style="bold blue")
    console.rule("[bold cyan]Training Management System[/bold cyan]")


def main():
    system = TrainingManagementSystem()
    show_banner()

    while True:
        if not system.logged_in_user:
            console.print("\n[bold blue]Main Menu[/bold blue]")
            console.print("1. Login\n2. Exit")
            choice = system.get_input("Enter choice (1-2): ")
            if choice is None:
                continue

            if choice == '1':
                if system.login():
                    if system.logged_in_user == "admin":
                        system.admin_menu()
                    else:
                        system.employee_menu()
            elif choice == '2':
                system.save_data()
                console.print("[blue]Goodbye![/blue]")
                break
            else:
                console.print("[red]Invalid choice.[/red]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[red]Program interrupted. Exiting...[/red]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        sys.exit(1)
