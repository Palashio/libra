#!/usr/bin/env python3
"""
Hello.py - A collection of random Python scripts demonstrating various functionality

This file contains examples of:
- Basic Python operations and data structures
- Data manipulation with pandas and numpy
- Machine learning examples with sklearn
- Data visualization with matplotlib and seaborn
- Basic libra usage examples
- File I/O operations
- String manipulation and list comprehensions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import random
import string
import json
import os
from datetime import datetime

# =============================================================================
# SECTION 1: Basic Python Operations and Data Structures
# =============================================================================

def basic_python_examples():
    """Demonstrate basic Python operations and data structures."""
    print("=== Basic Python Examples ===")
    
    # List comprehensions
    squares = [x**2 for x in range(10)]
    print(f"Squares: {squares}")
    
    # Dictionary comprehension
    word_lengths = {word: len(word) for word in ['hello', 'world', 'python', 'libra']}
    print(f"Word lengths: {word_lengths}")
    
    # Generator expression
    even_squares = (x**2 for x in range(20) if x % 2 == 0)
    print(f"Even squares: {list(even_squares)}")
    
    # String manipulation
    text = "Machine Learning with Libra"
    print(f"Original: {text}")
    print(f"Reversed: {text[::-1]}")
    print(f"Words: {text.split()}")
    print(f"Uppercase: {text.upper()}")
    
    # Working with sets
    set1 = {1, 2, 3, 4, 5}
    set2 = {4, 5, 6, 7, 8}
    print(f"Union: {set1 | set2}")
    print(f"Intersection: {set1 & set2}")
    print(f"Difference: {set1 - set2}")

def random_data_generator():
    """Generate random data for testing purposes."""
    print("\n=== Random Data Generator ===")
    
    # Generate random names
    first_names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia']
    
    people = []
    for _ in range(10):
        person = {
            'name': f"{random.choice(first_names)} {random.choice(last_names)}",
            'age': random.randint(18, 80),
            'salary': random.randint(30000, 120000),
            'department': random.choice(['Engineering', 'Marketing', 'Sales', 'HR'])
        }
        people.append(person)
    
    return people

# =============================================================================
# SECTION 2: Data Manipulation with Pandas and NumPy
# =============================================================================

def numpy_examples():
    """Demonstrate NumPy operations."""
    print("\n=== NumPy Examples ===")
    
    # Create arrays
    arr1 = np.random.rand(5, 3)
    arr2 = np.random.rand(5, 3)
    
    print(f"Array 1 shape: {arr1.shape}")
    print(f"Array 1:\n{arr1}")
    
    # Mathematical operations
    print(f"Element-wise addition:\n{arr1 + arr2}")
    print(f"Matrix multiplication:\n{np.dot(arr1.T, arr2)}")
    print(f"Mean along axis 0: {np.mean(arr1, axis=0)}")
    print(f"Standard deviation: {np.std(arr1)}")
    
    # Boolean indexing
    mask = arr1 > 0.5
    print(f"Values > 0.5: {arr1[mask]}")
    
    # Reshaping
    reshaped = arr1.reshape(-1)
    print(f"Reshaped to 1D: {reshaped.shape}")

def pandas_examples():
    """Demonstrate Pandas operations."""
    print("\n=== Pandas Examples ===")
    
    # Create DataFrame from random data
    people_data = random_data_generator()
    df = pd.DataFrame(people_data)
    
    print("Original DataFrame:")
    print(df.head())
    
    # Basic statistics
    print(f"\nDataFrame info:")
    print(df.describe())
    
    # Groupby operations
    dept_stats = df.groupby('department').agg({
        'age': ['mean', 'min', 'max'],
        'salary': ['mean', 'sum']
    })
    print(f"\nDepartment statistics:")
    print(dept_stats)
    
    # Filtering
    high_earners = df[df['salary'] > 70000]
    print(f"\nHigh earners (>70k): {len(high_earners)} people")
    
    # Adding new columns
    df['salary_category'] = pd.cut(df['salary'], 
                                  bins=[0, 50000, 80000, float('inf')], 
                                  labels=['Low', 'Medium', 'High'])
    print(f"\nSalary categories:")
    print(df['salary_category'].value_counts())
    
    return df

# =============================================================================
# SECTION 3: Machine Learning Examples with Scikit-learn
# =============================================================================

def classification_example():
    """Demonstrate a simple classification task."""
    print("\n=== Classification Example ===")
    
    # Generate synthetic classification data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                              n_redundant=10, n_clusters_per_class=1, random_state=42)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Classification accuracy: {accuracy:.4f}")
    
    # Feature importance
    feature_importance = clf.feature_importances_
    top_features = np.argsort(feature_importance)[-5:]
    print(f"Top 5 most important features: {top_features}")
    
    return clf, X_test, y_test

def regression_example():
    """Demonstrate a simple regression task."""
    print("\n=== Regression Example ===")
    
    # Generate synthetic regression data
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Linear Regression model
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    
    # Make predictions
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {reg.score(X_test, y_test):.4f}")
    
    return reg, X_test, y_test, y_pred

# =============================================================================
# SECTION 4: Data Visualization Examples
# =============================================================================

def create_visualizations():
    """Create various types of plots using matplotlib and seaborn."""
    print("\n=== Creating Visualizations ===")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Random Python Scripts - Data Visualizations', fontsize=16)
    
    # Plot 1: Line plot
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    axes[0, 0].plot(x, y1, label='sin(x)', linewidth=2)
    axes[0, 0].plot(x, y2, label='cos(x)', linewidth=2)
    axes[0, 0].set_title('Trigonometric Functions')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Histogram
    data = np.random.normal(100, 15, 1000)
    axes[0, 1].hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].set_title('Normal Distribution Histogram')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency')
    
    # Plot 3: Scatter plot
    x_scatter = np.random.randn(200)
    y_scatter = 2 * x_scatter + np.random.randn(200) * 0.5
    colors = np.random.rand(200)
    axes[1, 0].scatter(x_scatter, y_scatter, c=colors, alpha=0.6, cmap='viridis')
    axes[1, 0].set_title('Scatter Plot with Color Mapping')
    axes[1, 0].set_xlabel('X values')
    axes[1, 0].set_ylabel('Y values')
    
    # Plot 4: Bar plot
    categories = ['A', 'B', 'C', 'D', 'E']
    values = [23, 45, 56, 78, 32]
    bars = axes[1, 1].bar(categories, values, color=['red', 'green', 'blue', 'orange', 'purple'])
    axes[1, 1].set_title('Bar Chart Example')
    axes[1, 1].set_ylabel('Values')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                       str(value), ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(f'{output_dir}/hello_visualizations.png', dpi=300, bbox_inches='tight')
    print(f"Visualizations saved to {output_dir}/hello_visualizations.png")
    plt.close()

def seaborn_examples():
    """Create visualizations using seaborn."""
    print("\n=== Seaborn Examples ===")
    
    # Create sample data
    df = pandas_examples()
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Box plot
    sns.boxplot(data=df, x='department', y='salary', ax=axes[0])
    axes[0].set_title('Salary Distribution by Department')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Correlation heatmap (for numeric columns only)
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0, ax=axes[1])
    axes[1].set_title('Correlation Matrix')
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(f'{output_dir}/seaborn_examples.png', dpi=300, bbox_inches='tight')
    print(f"Seaborn examples saved to {output_dir}/seaborn_examples.png")
    plt.close()

# =============================================================================
# SECTION 5: File I/O Operations
# =============================================================================

def file_io_examples():
    """Demonstrate file input/output operations."""
    print("\n=== File I/O Examples ===")
    
    # Create output directory
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Write text file
    sample_text = """This is a sample text file created by hello.py
It contains multiple lines of text.
Each line demonstrates different aspects of file I/O.
Python makes file operations very easy!
"""
    
    text_file = f'{output_dir}/sample_text.txt'
    with open(text_file, 'w') as f:
        f.write(sample_text)
    print(f"Text file written to {text_file}")
    
    # Read and process the text file
    with open(text_file, 'r') as f:
        lines = f.readlines()
        word_count = sum(len(line.split()) for line in lines)
        char_count = sum(len(line) for line in lines)
    
    print(f"File statistics: {len(lines)} lines, {word_count} words, {char_count} characters")
    
    # Write JSON file
    sample_data = {
        'timestamp': datetime.now().isoformat(),
        'project': 'Libra',
        'description': 'Ergonomic machine learning library',
        'features': ['Neural Networks', 'SVM', 'Decision Trees', 'Data Visualization'],
        'stats': {
            'lines_of_code': 5000,
            'contributors': 10,
            'stars': 500
        }
    }
    
    json_file = f'{output_dir}/sample_data.json'
    with open(json_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    print(f"JSON file written to {json_file}")
    
    # Read JSON file
    with open(json_file, 'r') as f:
        loaded_data = json.load(f)
    print(f"Loaded JSON data: {loaded_data['project']} - {loaded_data['description']}")
    
    # Write CSV file using pandas
    people_data = random_data_generator()
    df = pd.DataFrame(people_data)
    csv_file = f'{output_dir}/people_data.csv'
    df.to_csv(csv_file, index=False)
    print(f"CSV file written to {csv_file}")
    
    # Read CSV file
    loaded_df = pd.read_csv(csv_file)
    print(f"Loaded CSV shape: {loaded_df.shape}")

# =============================================================================
# SECTION 6: Basic Libra Usage Example
# =============================================================================

def libra_usage_example():
    """Demonstrate basic libra usage (conceptual example)."""
    print("\n=== Libra Usage Example (Conceptual) ===")
    
    # Note: This is a conceptual example showing how libra would be used
    # In a real scenario, you would need actual data files
    
    print("# Example of how to use Libra for machine learning:")
    print("from libra import client")
    print("")
    print("# Create a client object with your dataset")
    print("ml_client = client('path/to/your/dataset.csv')")
    print("")
    print("# Perform various ML queries")
    print("ml_client.neural_network_query('predict house prices')")
    print("ml_client.svm_query('classify customer segments')")
    print("ml_client.decision_tree_query('predict loan approval')")
    print("")
    print("# Analyze results")
    print("ml_client.analyze()")
    print("")
    print("# Get model information")
    print("model_info = ml_client.info()")
    print("accuracy = ml_client.models['neural_network']['accuracy']")
    print("")
    print("# Generate plots")
    print("plots = ml_client.models['neural_network']['plots']")
    
    # Create a mock dataset that could be used with libra
    mock_dataset = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'target': np.random.choice([0, 1], 100)
    })
    
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    mock_file = f'{output_dir}/mock_dataset.csv'
    mock_dataset.to_csv(mock_file, index=False)
    print(f"\nMock dataset created at {mock_file} for libra experimentation")
    print(f"Dataset shape: {mock_dataset.shape}")
    print("You can use this dataset with libra by running:")
    print(f"client('{mock_file}').neural_network_query('predict target')")

# =============================================================================
# SECTION 7: Advanced Python Concepts
# =============================================================================

def advanced_python_examples():
    """Demonstrate advanced Python concepts."""
    print("\n=== Advanced Python Examples ===")
    
    # Decorators
    def timing_decorator(func):
        """A decorator to measure function execution time."""
        import time
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
            return result
        return wrapper
    
    @timing_decorator
    def slow_function():
        """A function that takes some time to execute."""
        import time
        time.sleep(0.1)
        return sum(range(1000000))
    
    result = slow_function()
    print(f"Result: {result}")
    
    # Context managers
    class CustomContextManager:
        def __enter__(self):
            print("Entering context")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            print("Exiting context")
            return False
    
    with CustomContextManager() as cm:
        print("Inside context manager")
    
    # Generators
    def fibonacci_generator(n):
        """Generate fibonacci numbers up to n."""
        a, b = 0, 1
        count = 0
        while count < n:
            yield a
            a, b = b, a + b
            count += 1
    
    fib_numbers = list(fibonacci_generator(10))
    print(f"First 10 Fibonacci numbers: {fib_numbers}")
    
    # Lambda functions and functional programming
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Filter even numbers
    evens = list(filter(lambda x: x % 2 == 0, numbers))
    print(f"Even numbers: {evens}")
    
    # Map to squares
    squares = list(map(lambda x: x**2, numbers))
    print(f"Squares: {squares}")
    
    # Reduce to sum
    from functools import reduce
    total = reduce(lambda x, y: x + y, numbers)
    print(f"Sum using reduce: {total}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run all examples."""
    print("🐍 Welcome to hello.py - Random Python Scripts Collection! 🐍")
    print("=" * 60)
    
    try:
        # Run all examples
        basic_python_examples()
        numpy_examples()
        pandas_examples()
        classification_example()
        regression_example()
        create_visualizations()
        seaborn_examples()
        file_io_examples()
        libra_usage_example()
        advanced_python_examples()
        
        print("\n" + "=" * 60)
        print("🎉 All examples completed successfully!")
        print("Check the 'output' directory for generated files and visualizations.")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        print("Some examples may require additional libraries to be installed.")
        print("Try running: pip install numpy pandas matplotlib seaborn scikit-learn")

if __name__ == "__main__":
    main()
