import os
import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Directory path and file path
directory_path = r'C:\Users\HP\Desktop\Slashmark\basic'
file_path = os.path.join(directory_path, 'tasks.csv')

# Ensure the directory exists
os.makedirs(directory_path, exist_ok=True)

# Load pre-existing tasks from a CSV file (if any)
try:
    tasks = pd.read_csv(file_path)
except FileNotFoundError:
    tasks = pd.DataFrame(columns=['description', 'priority'])

# Save the CSV Task File
def save_tasks():
    tasks.to_csv(file_path, index=False)

# Train the task priority classifier
if not tasks.empty:
    vectorizer = CountVectorizer()
    clf = MultinomialNB()
    model = make_pipeline(vectorizer, clf)
    model.fit(tasks['description'], tasks['priority'])

# Add Task
def add_task(description, priority):
    global tasks
    new_task = pd.DataFrame({'description': [description], 'priority': [priority]})
    tasks = pd.concat([tasks, new_task], ignore_index=True)
    save_tasks()

# Remove Task
def remove_task(description):
    global tasks
    tasks = tasks[tasks['description'] != description]
    save_tasks()

# List All Tasks
def list_tasks():
    if tasks.empty:
        print("No tasks available.")
    else:
        print(tasks)

# Recommend Tasks to the User
def recommend_task():
    if not tasks.empty:
        high_priority_tasks = tasks[tasks['priority'] == 'High']
        if not high_priority_tasks.empty:
            descriptions = high_priority_tasks['description'].tolist()
            random_task = random.choice(descriptions)
            print(f"Recommended task: {random_task} - Priority: High")
        else:
            print("No high-priority tasks available for recommendation.")
    else:
        print("No tasks available for recommendations.")

# MENU
while True:
    print("\nTask Management App")
    print("1. Add Task")
    print("2. Remove Task")
    print("3. List Tasks")
    print("4. Recommend Task")
    print("5. Exit")

    choice = input("Select an option: ")

    if choice == "1":
        description = input("Enter task description: ")
        priority = input("Enter task priority (Low/Medium/High): ").capitalize()
        add_task(description, priority)
        print("Task added successfully.")

    elif choice == "2":
        description = input("Enter task description to remove: ")
        remove_task(description)
        print("Task removed successfully.")

    elif choice == "3":
        list_tasks()

    elif choice == "4":
        recommend_task()

    elif choice == "5":
        print("Goodbye!")
        break

    else:
        print("Invalid option. Please select a valid option.")
