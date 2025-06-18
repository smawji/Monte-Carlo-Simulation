import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import triang


# Read and validate the Excel file
def read_and_validate_data(filename):
    try:
        df = pd.read_csv(filename, index_col=0)

        # Check if required rows exist
        if not all(x in df.index for x in ['Opt', 'ML', 'Pess']):
            raise ValueError("Missing required rows (Opt, ML, Pess)")

        # Transpose to have tasks as columns
        df = df.T

        # Validate numerical values
        for col in ['Opt', 'ML', 'Pess']:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Non-numeric values found in {col} row")

        df.to_csv('critical_path_clean.csv')
        return df

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None


# Calculate PERT durations and ranges
def calculate_pert(data):
    # Calculate PERT: (O + 4*ML + P) / 6
    data['PERT'] = (data['Opt'] + 4 * data['ML'] + data['Pess']) / 6

    # Calculate ranges
    data['Min'] = data['Opt']
    data['Max'] = data['Pess']

    # Calculate total project durations
    totals = {
        'Optimistic': data['Opt'].sum(),
        'Most Likely': data['ML'].sum(),
        'Pessimistic': data['Pess'].sum(),
        'PERT': data['PERT'].sum()
    }

    # Save summary
    summary = pd.DataFrame([totals], index=['Total Duration'])
    data.to_csv('pert_summary.csv')
    summary.to_csv('pert_summary.csv', mode='a')  # Append totals

    return data, totals


# Monte Carlo simulation
def monte_carlo_simulation(data, num_simulations=1000):
    tasks = data.index
    simulations = pd.DataFrame()

    # Generate random durations for each task
    for task in tasks:
        opt, ml, pess = data.loc[task, ['Opt', 'ML', 'Pess']]

        # Handle case where Opt == Pess
        if opt == pess:
            # If all three estimates are equal
            samples = np.full(num_simulations, opt)
        else:
            # Standard triangular distribution
            samples = np.random.triangular(opt, ml, pess, num_simulations)

        simulations[task] = samples

    # Calculate total project duration for each simulation
    simulations['Total'] = simulations.sum(axis=1)
    simulations.to_csv('monte_carlo_raw.csv', index=False)

    return simulations


# Create histogram for Task1
def plot_task1_histogram(data, simulations):
    task1_data = data.loc['Task1', ['Opt', 'ML', 'Pess']]
    samples = simulations['Task1']

    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=30, density=True, alpha=0.7, label='Sampled Durations')

    # Plot
    opt, ml, pess = task1_data
    x = np.linspace(opt, pess, 100)
    c = (ml - opt) / (pess - opt)  # Mode position
    scale = pess - opt
    plt.plot(x, triang.pdf(x, c, loc=opt, scale=scale),
             'r-', lw=2, label='Triangular Distribution')

    plt.title('Task1 Duration Distribution')
    plt.xlabel('Duration')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.savefig('task1_histogram.png')
    plt.close()


# Plot confidence curve
def generate_confidence_curve(simulations):
    total_durations = simulations['Total']
    percentiles = np.arange(60, 100, 0.1)
    confidence_values = np.percentile(total_durations, percentiles)

    # Create confidence table
    confidence_table = pd.DataFrame({
        'Percentile': percentiles,
        'Duration': confidence_values
    })
    confidence_table.to_csv('confidence_curve.csv', index=False)

    # Plot confidence curve
    plt.figure(figsize=(10, 6))
    plt.plot(percentiles, confidence_values, 'b-')
    plt.title('Project Duration Confidence Curve')
    plt.xlabel('Confidence Level (%)')
    plt.ylabel('Project Duration')
    plt.grid(True)
    plt.savefig('confidence_plot.png')
    plt.close()

    return confidence_table

def main():
    # Step 1: Read and validate data
    data = read_and_validate_data('Critical Path Data.csv')
    if data is None:
        return

    # Step 2: Calculate PERT estimates
    pert_data, totals = calculate_pert(data)
    print("PERT Summary:")
    print(pd.DataFrame([totals]))

    # Step 3: Monte Carlo simulation
    simulations = monte_carlo_simulation(pert_data)

    # Step 4: Task1 histogram
    plot_task1_histogram(pert_data, simulations)
    print("\nTask1 histogram saved as task1_histogram.png")

    # Step 5: Confidence curve
    generate_confidence_curve(simulations)

    # Answer management questions
    confidence_levels = [70, 80, 90]
    answers = {
        f"{level}% confidence duration":
            np.percentile(simulations['Total'], level)
        for level in confidence_levels
    }

    print("\nManagement Questions Answers:")
    for level, duration in answers.items():
        print(f"{level}: {duration:.2f} days")

    # Save answers to file
    with open('confidence_answers.txt', 'w') as f:
        f.write("Minimum project durations at given confidence levels:\n")
        for level, duration in answers.items():
            f.write(f"{level}: {duration:.2f} days\n")


if __name__ == "__main__":
    main()