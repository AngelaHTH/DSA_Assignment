# Angela Ho Tian Hui 231058z
# Admin login is username: admin, password: admin123
import re
import shelve
import pandas as pd
from fpdf import FPDF
from datetime import datetime
from rich.console import Console
from rich.table import Table
from tabulate import tabulate
import pyfiglet
import sys
import logging
from logging.handlers import RotatingFileHandler
import os
import getpass  # For secure password input


# Initialize rich console with wider display
console = Console(width=140)  # Increased width to prevent truncation


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


class Employee:
    def __init__(self, name, employee_id, email, department, full_time_status, enrolled_programmes=None):
        self.name = name
        self.employee_id = employee_id
        self.email = email.lower()  # Store email in lowercase
        self.department = department
        self.full_time_status = full_time_status
        self.enrolled_programmes = enrolled_programmes if enrolled_programmes else []
        self.password = None
        self.feedback = []
        self.pending_requests = []

    def add_training_programme(self, programme):
        programme_lower = programme.lower()
        if not any(p.lower() == programme_lower for p in self.enrolled_programmes):
            self.enrolled_programmes.append(programme)
            action_logger.info(f"Employee {self.employee_id} enrolled in {programme}")
            return True
        return False

    def remove_training_programme(self, programme):
        programme_lower = programme.lower()
        for i, prog in enumerate(self.enrolled_programmes):
            if prog.lower() == programme_lower:
                self.enrolled_programmes.pop(i)
                action_logger.info(f"Employee {self.employee_id} removed from {programme}")
                return True
        return False

    def display_details(self):
        status = "Full-time" if self.full_time_status else "Part-time"
        programmes = "\n".join(self.enrolled_programmes) if self.enrolled_programmes else "None"

        table = Table(title=f"Employee Details - {self.name}", show_header=True, header_style="bold blue", expand=True)
        table.add_column("Field", style="dim", width=20)
        table.add_column("Value", width=60)

        table.add_row("Employee ID", str(self.employee_id))
        table.add_row("Name", self.name)
        table.add_row("Email", self.email)
        table.add_row("Department", self.department)
        table.add_row("Status", status)
        table.add_row("Programmes", programmes)

        console.print(table)

    def add_feedback(self, programme, feedback_text, rating):
        feedback_entry = {
            'programme': programme,
            'feedback': feedback_text,
            'rating': rating,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.feedback.append(feedback_entry)
        action_logger.info(f"Employee {self.employee_id} submitted feedback for {programme}")

    def request_enrollment(self, programme):
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
            'pending_requests': self.pending_requests
        }


class TrainingManagementSystem:
    def __init__(self):
        self.employees = {}
        self.logged_in_user = None
        self.admin_username = "admin"
        self.admin_password = "admin123"
        self.load_data()
        system_logger.info("System initialized")

    def save_data(self):
        with shelve.open('employee_data') as db:
            db['employees'] = {eid: emp.to_dict() for eid, emp in self.employees.items()}
        system_logger.info("Data saved")

    def load_data(self):
        try:
            with shelve.open('employee_data') as db:
                if 'employees' in db:
                    employees_data = db['employees']
                    for eid, emp_data in employees_data.items():
                        employee = Employee(
                            emp_data['name'],
                            int(eid),
                            emp_data['email'],
                            emp_data['department'],
                            emp_data['full_time_status'],
                            emp_data['enrolled_programmes']
                        )
                        employee.password = emp_data.get('password')
                        employee.feedback = emp_data.get('feedback', [])
                        employee.pending_requests = emp_data.get('pending_requests', [])
                        self.employees[int(eid)] = employee
            system_logger.info("Data loaded")
        except Exception as e:
            system_logger.error(f"Error loading data: {str(e)}")
            self.employees = {}

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

        # Department - removed restrictions
        department = self.get_input("Enter department: ", required=True)
        if department is None:
            return
        department = department.title()  # Just format nicely

        # Status
        while True:
            status = self.get_input("Full-time? (Y/N): ", required=True)
            if status is None:
                return
            status = status.upper()
            if status in ['Y', 'N']:
                full_time = status == 'Y'
                break
            console.print("[red]Please enter Y or N.[/red]")

        new_employee = Employee(name, employee_id, email, department, full_time)
        self.employees[employee_id] = new_employee
        self.save_data()
        action_logger.info(f"Added employee: {name} (ID: {employee_id})")
        console.print(f"\n[green]Employee {name} added successfully![/green]")

    def get_input(self, prompt, required=False, password=False, allow_back=True):
        while True:
            try:
                if password:
                    user_input = getpass.getpass(prompt)
                else:
                    print(prompt, end='', flush=True)  # Explicitly print prompt
                    user_input = input().strip()

                if allow_back and (user_input.lower() == 'back' or user_input == '0'):
                    return None
                if required and not user_input:
                    console.print("[red]This field is required.[/red]")
                    continue
                return user_input
            except EOFError:
                return None
            except KeyboardInterrupt:
                console.print("\n[red]Operation cancelled.[/red]")
                return None

    def display_employee_table(self, employees, extra_columns=None):
        """Helper method to display employee data in consistent tabulate format"""
        # Prepare table data with all information
        table_data = []
        for emp in employees:
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
                emp.email,
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

        # Define base headers
        headers = [
            "ID", "Name", "Email", "Status", "Department",
            "Programmes", "Programmes", "Badges", "Points", "Badge Names"
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
        self.display_employee_table([employee])
        action_logger.info(f"Displayed employee {employee_id} for enrollment")

        programme = self.get_input("Enter programme name: ", required=True)
        if not programme:
            return

        if employee.add_training_programme(programme):
            self.save_data()
            action_logger.info(f"Enrolled employee {employee_id} in {programme}")
            console.print(f"\n[green]{employee.name} enrolled in {programme}.[/green]")
            self.display_employee_table([employee])
        else:
            console.print(f"\n[yellow]{employee.name} is already enrolled in this programme.[/yellow]")
            action_logger.info(f"Enrollment failed - already enrolled: {employee_id} in {programme}")

    def unenroll_programme(self):
        console.print("[italic]Enter 'back' or '0' at any time to cancel[/italic]\n")
        employee_id = self.select_employee()
        if not employee_id:
            return

        employee = self.employees[employee_id]
        if not employee.enrolled_programmes:
            console.print("[yellow]Employee is not enrolled in any programmes.[/yellow]")
            return

        console.print("\n[bold]Current Programmes:[/bold]")
        for i, prog in enumerate(employee.enrolled_programmes, 1):
            console.print(f"{i}. {prog}")

        prog_choice = self.get_input("Select programme to unenroll (number): ", required=True)
        if prog_choice is None: return

        try:
            prog_choice = int(prog_choice) - 1
            if 0 <= prog_choice < len(employee.enrolled_programmes):
                programme = employee.enrolled_programmes[prog_choice]
                if employee.remove_training_programme(programme):
                    self.save_data()
                    console.print(f"\n[green]Successfully unenrolled {employee.name} from {programme}.[/green]")
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
        console.print("1. Name\n2. Employee ID\n3. Email\n4. Department\n5. Status\n6. Cancel")
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
        console.print("1. Admin\n2. Employee\n3. Exit")

        while True:
            choice = self.get_input("Enter choice (1-3): ", required=True)
            if choice is None:
                continue

            try:
                choice = int(choice)
            except ValueError:
                console.print("[red]Please enter a valid number (1-3)[/red]")
                continue

            if choice == 1:
                # Admin login
                username = self.get_input("Username: ", required=True)
                if username is None:
                    continue

                password = self.get_input("Password: ", required=True)
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

            elif choice == 2:
                # Employee login
                employee_id = self.select_employee()
                if not employee_id:
                    console.print("[red]Invalid Employee ID.[/red]")
                    return False

                employee = self.employees[employee_id]

                if employee.password is None:
                    console.print("\n[yellow]First-time login. Set your password.[/yellow]")
                    while True:
                        password = self.get_input("Set password: ", required=True)
                        if password is None:
                            break

                        confirm = self.get_input("Confirm password: ", required=True)
                        if confirm is None:
                            break

                        if password == confirm:
                            employee.password = password
                            self.save_data()
                            self.logged_in_user = employee_id
                            action_logger.info(f"Employee {employee_id} set password")
                            console.print("\n[green]Password set and login successful![/green]")
                            return True
                        else:
                            console.print("[red]Passwords do not match. Try again.[/red]")
                else:
                    password = self.get_input("Password: ", required=True)
                    if password is None:
                        continue

                    if password == employee.password:
                        self.logged_in_user = employee_id
                        action_logger.info(f"Employee {employee_id} logged in")
                        console.print("\n[green]Login successful![/green]")
                        return True
                    else:
                        console.print("[red]Invalid password.[/red]")
                        return False

            elif choice == 3:
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
        except:
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
        if badges:
            console.print("  " + ", ".join(badges))
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
        programme = self.get_input("Programme name to request: ", required=True)
        if programme is None:
            return

        if employee.request_enrollment(programme):
            self.save_data()
            console.print(f"\n[green]Request submitted for {programme}.[/green]")
        else:
            console.print(f"\n[yellow]Already enrolled or pending for {programme}.[/yellow]")

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

    def admin_menu(self):
        while True:
            console.print("\n[bold blue]Admin Menu[/bold blue]")
            console.print("0. Back to Main Menu\n1. View Employees\n2. Add Employee\n3. Modify Employee")
            console.print("4. Delete Employee\n5. Enroll Employee\n6. Unenroll Employee")
            console.print("7. Sort by Department\n8. Sort by Programmes\n9. Sort by Status")
            console.print("10. Search Employee\n11. Filter by Programme\n12. Pending Requests")
            console.print("13. View Feedback\n14. Manage Badges\n15. Export Data\n16. Logout\n17. Exit")

            choice = self.get_input("Enter choice (1-17): ")
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
                self.bubble_sort_by_department()
            elif choice == '8':
                self.selection_sort_by_programmes()
            elif choice == '9':
                self.sort_by_status()
            elif choice == '10':
                self.search_employee()
            elif choice == '11':
                self.filter_by_programme()
            elif choice == '12':
                self.view_pending_requests()
            elif choice == '13':
                self.view_feedback()
            elif choice == '14':
                self.manage_badges()
            elif choice == '15':
                self.export_data()
            elif choice == '16':
                action_logger.info("Admin logged out")
                self.logged_in_user = None
                console.print("[green]Logged out.[/green]")
                break
            elif choice == '17':
                self.save_data()
                console.print("\n[blue]Goodbye![/blue]")
                sys.exit(0)
            else:
                console.print("[red]Invalid choice.[/red]")

    def employee_menu(self):
        """Menu for regular employees after login"""
        employee = self.employees[self.logged_in_user]

        while True:
            console.print(f"\n[bold blue]Employee Menu - {employee.name}[/bold blue]")
            console.print("1. View Employee Details\n2. Submit Feedback\n3. Request Programme Enrollment")
            console.print("4. View Badges\n5. Change Password\n6. Logout")

            choice = self.get_input("Enter choice (1-6): ")
            if choice is None:
                continue

            if choice == '1':
                self.show_employee_details(employee)
            elif choice == '2':
                self.submit_feedback(employee)
            elif choice == '3':
                self.request_enrollment(employee)
            elif choice == '4':
                self.view_employee_badges(employee)
            elif choice == '5':
                self.change_password(employee)
            elif choice == '6':
                action_logger.info(f"Employee {employee.employee_id} logged out")
                self.logged_in_user = None
                console.print("[green]Logged out successfully![/green]")
                break
            else:
                console.print("[red]Invalid choice.[/red]")

    def view_employee_badges(self, employee):
        """Display badges for the logged-in employee"""
        try:
            with shelve.open('badges_db', flag='r') as badges_db:
                badges = badges_db.get(str(employee.employee_id), [])

                console.print("\n[bold blue]Your Badges[/bold blue]")
                if not badges:
                    console.print("[yellow]No badges earned yet.[/yellow]")
                    return  # Fix: This return was causing the function to exit early

                table = Table(show_header=True, header_style="bold blue", show_lines=True)
                table.add_column("Badge", style="cyan")
                table.add_column("Points", style="green")

                for badge in badges:
                    points = self.calculate_points([badge])  # Calculate points for this single badge
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

        console.print(tabulate(table_data, headers=headers, tablefmt="fancy_grid",colalign=colalign))

    def export_data(self):
        """Export employee data with badges information"""
        console.print("\n[bold blue]Export Employee Data[/bold blue]")

        try:
            # Prepare data and sort by employee ID
            table_data = []
            with shelve.open('badges_db', flag='r') as badges_db:
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
            action = self.get_input("\nEnter 'approve ID' or 'reject ID' (or 'back'): ")
            if action is None or action.lower() == 'back':
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
