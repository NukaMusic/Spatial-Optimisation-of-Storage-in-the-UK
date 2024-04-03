import os

os.chdir('/Users/nukamila/Library/CloudStorage/OneDrive-UniversityofEdinburgh/Year 5 Individual Project/Code/PyPSA-GB/PyPSA-GB')

import pypsa
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
plt.style.use('ggplot')
import numpy as np
import pandas as pd
import data_reader_writer
import cartopy


'''Setup parameters for the solver'''
# Set the required inputs for the LOPF: the start, end and year of simulation, and the timestep.
# write csv files for import
start = '2035-01-01 00:00:00'
end = ('2035-07-31 23:00:00')
year = int(start[0:4])

# time step
time_step = 1.

# Choose future energy scenario
scenario = 'Leading The Way'
# scenario = 'Consumer Transformation'
# scenario = 'System Transformation'
# scenario = 'Steady Progression'

# Choose baseline year
year_baseline = 2012

data_reader_writer.data_writer(start, end, time_step, year, demand_dataset='eload', year_baseline=year_baseline,
                               scenario=scenario, FES=2022, merge_generators=True, scale_to_peak=True,
                               networkmodel='Reduced', P2G=True)
'''End setup'''


def run_optimization(network):
    # Run the network optimization
    network.optimize(network.snapshots, solver_name="gurobi")


def plot_storage_technology_capacity(network, storage_type, plot_extent=None, marker_scaler=0.2):
    # Ensure plot_extent has a default value if not provided
    if plot_extent is None:
        plot_extent = [-7, 2, 49, 59]  # Default to a sample extent

    # Read the buses.csv to get the coordinates
    df_buses = pd.read_csv('../data/network/buses.csv', index_col=0)

    # Filter for specified storage units and directly merge with df_buses on 'bus' column
    storage_units = network.storage_units[network.storage_units.carrier == storage_type].merge(
        df_buses, left_on='bus', right_index=True, how='left')

    # Extract 'lon' and 'lat' values and 'p_nom' for storage capacity
    lon, lat, sizes = storage_units['x'].values, storage_units['y'].values, storage_units['p_nom'].values

    # Adjust marker size for better visibility
    sizes_scaled = sizes * marker_scaler

    # Plot setup
    plt.figure(figsize=(8, 10))
    ax = plt.axes(projection=cartopy.crs.PlateCarree())
    ax.set_extent(plot_extent, crs=cartopy.crs.PlateCarree())

    # Adding map features
    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
    ax.add_feature(cartopy.feature.OCEAN)

    # Plotting storage locations
    color = 'deepskyblue' if storage_type == 'Battery' else 'limegreen'
    scatter = ax.scatter(lon, lat, s=sizes_scaled, color=color, edgecolor='black', alpha=0.7,
                         transform=cartopy.crs.PlateCarree(), label=f'{storage_type} Storage Capacity')

    # Dynamically adjust legend sizes
    min_size = np.min(sizes)
    max_size = np.max(sizes)
    mean_size = (min_size + max_size) / 2
    legend_sizes = [min_size, mean_size, max_size]
    legend_labels = [f'{size:.2f} MW' for size in legend_sizes]

    legend_handles = [mlines.Line2D([], [], color='w', markerfacecolor=color, markeredgecolor='black',
                                    markersize=np.log(size), marker='o', linestyle='None', label=label)
                      for size, label in zip(legend_sizes, legend_labels)]
    legend = ax.legend(handles=legend_handles, title="Storage Capacity", loc="upper left", fontsize='small')
    ax.add_artist(legend)

    plt.title(f'{storage_type} Storage Capacity Across Nodes', fontsize=16)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.tight_layout()

    plt.show()


def plot_curtailment(network):
    # Get data on capacities
    p_by_carrier = network.generators_t.p.groupby(network.generators.carrier, axis=1).sum()

    # Plot curtailment data
    carriers = ["Wind Onshore", "Wind Offshore", "Solar Photovoltaics"]

    for carrier in carriers:
        capacity = network.generators.groupby("carrier").sum().at[carrier, "p_nom"]
        p_available = network.generators_t.p_max_pu.multiply(network.generators["p_nom"])
        p_available_by_carrier = p_available.groupby(network.generators.carrier, axis=1).sum()
        p_curtailed_by_carrier = p_available_by_carrier - p_by_carrier
        p_df = pd.DataFrame({carrier + " available": p_available_by_carrier[carrier],
                             carrier + " dispatched": p_by_carrier[carrier],
                             carrier + " curtailed": p_curtailed_by_carrier[carrier]})


        p_df[carrier + " capacity"] = capacity
        p_df[carrier + " curtailed"][p_df[carrier + " curtailed"] < 0.] = 0.
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(15, 10)
        p_df[[carrier + " dispatched", carrier + " curtailed"]].plot(kind="area", ax=ax, linewidth=0, color=["cornflowerblue", "lightcoral"])

        plt.title(f'{carrier} Curtailment', fontsize=16)
        ax.set_xlabel("Date and Time")
        ax.set_ylabel("Power [MW]")
        ax.legend()
        plt.tight_layout()
        plt.show()

        # Calculate and output the percentage of curtailed energy
        total_curtailment = p_df[carrier + " curtailed"].sum()
        total_available = p_df[carrier + " available"].sum()
        percentage_curtailment = (total_curtailment / total_available) * 100 if total_available > 0 else 0
        print(f"{carrier} - Percentage of curtailed energy over the period: {percentage_curtailment:.2f}%")
        print(f"{carrier} - Total energy curtailed: {total_curtailment:.2f} MWh")


def calculate_specific_storage_capacity(network, storage_type):
    """Calculate the total installed storage capacity for a specific storage type."""
    if storage_type == "Battery":
        # Filter by carrier type if your storage units have a 'carrier' attribute.
        return sum(network.storage_units.loc[network.storage_units.carrier == 'Battery', 'p_nom'])
    elif storage_type == "P2G":
        # Adjust the condition for filtering P2G units as per your network's configuration.
        return sum(network.storage_units.loc[network.storage_units.carrier == 'P2G', 'p_nom'])


def print_storage_data_by_type(network, storage_type=None):
    """
    Prints the details of storage units in the network.
    If storage_type is specified, it filters the storage units by the given type.

    :param network: The network containing storage units.
    :param storage_type: The type of storage to filter by (optional).
    """
    for storage_unit in network.storage_units.iterrows():
        # Each 'storage_unit' is a tuple (index, data) where data contains all the properties
        unit_name, attributes = storage_unit[0], storage_unit[1]

        # If a storage_type is specified, only print units of that type
        if storage_type and attributes['carrier'] != storage_type:
            continue

        print(f"Storage Unit Name: {unit_name}")
        for attr, value in attributes.items():
            print(f"  {attr}: {value}")
        print()  # Print a newline for better readability between units


def update_storage_units(network, updates):
    """
    Updates the 'p_nom' and 'state_of_charge_initial' of storage units based on a given updates dictionary.

    Parameters:
    - network: The network object containing the storage units.
    - updates: A dictionary where keys are the names of the storage units and values are the new capacities (p_nom).

    Each storage unit's 'state_of_charge_initial' will be updated to be no more than p_nom * max_hours.
    """
    for storage_unit_id, new_p_nom in updates.items():
        # Check if the storage unit exists in the network
        if storage_unit_id in network.storage_units.index:
            # Update p_nom
            network.storage_units.at[storage_unit_id, 'p_nom'] = new_p_nom*2

            # Calculate the maximum allowable state_of_charge_initial
            max_hours = network.storage_units.at[storage_unit_id, 'max_hours']
            max_initial_soc = new_p_nom * max_hours

            # Update state_of_charge_initial if necessary
            current_initial_soc = network.storage_units.at[storage_unit_id, 'state_of_charge_initial']
            if current_initial_soc > max_initial_soc:
                network.storage_units.at[storage_unit_id, 'state_of_charge_initial'] = max_initial_soc
        else:
            print(f"Storage unit {storage_unit_id} not found in the network.")


def calculate_tot_marginal_price(network):
    """
    Calculate the average system-wide marginal price of electricity across all snapshots,
    excluding values over £1,000,000, and plot the marginal price for each snapshot,
    connecting points close to excluded values.
    """
    lmps = network.buses_t.marginal_price  # DataFrame of LMPs for all snapshots
    loads = network.loads_t.p_set  # DataFrame of loads for all snapshots

    # Ensure alignment between loads and LMPs
    aligned_loads = loads.reindex(lmps.index, fill_value=0)

    # Calculate total load per snapshot
    total_load_per_snapshot = aligned_loads.sum(axis=1)

    # Filter out snapshots where total load is zero or negative
    valid_snapshots = total_load_per_snapshot > 0
    if not valid_snapshots.any():
        print("Error: No valid snapshots with positive total load.")
        return None

    # Compute weighted average LMPs for valid snapshots
    weighted_lmps = (lmps[valid_snapshots] * aligned_loads[valid_snapshots]).sum(axis=1) / total_load_per_snapshot[
        valid_snapshots]

    # Exclude values over £1,000,000 from calculation
    reasonable_values = weighted_lmps <= 1e6
    adjusted_weighted_lmps = weighted_lmps[reasonable_values]

    # Calculate the average of system-wide marginal prices across valid snapshots, excluding extreme values
    average_system_wide_marginal_price = adjusted_weighted_lmps.mean()

    # Plotting: Replace extreme values with NaN for plotting
    weighted_lmps_for_plot = weighted_lmps.copy()
    weighted_lmps_for_plot[~reasonable_values] = np.nan

    # Interpolate to connect points on either side of NaNs for visualization
    interpolated_for_plot = weighted_lmps_for_plot.interpolate()

    plt.figure(figsize=(10, 6))
    interpolated_for_plot.plot()
    plt.title('Marginal Price Over Time Period')
    plt.xlabel('Date and Time')
    plt.ylabel('Marginal Price (£/MWh)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Average system-wide marginal price (excluding >£1M): £{average_system_wide_marginal_price:.2f}/MWh")
    return average_system_wide_marginal_price


def plot_line_loading(network):
    """
    Plots both the maximum and average line loading across all snapshots for each line in the network,
    with legends indicating the loading levels. It also prints the average line loading for the entire network.
    """
    # Calculate maximum and average loading for each line as a fraction of its nominal capacity
    max_loading = network.lines_t.p0.divide(network.lines.s_nom).abs().max()
    avg_loading_per_line = network.lines_t.p0.divide(network.lines.s_nom).abs().mean()

    # Calculate the average loading for the entire network
    avg_loading_network = avg_loading_per_line.mean()

    # Define colormap
    cmap = plt.cm.jet

    # Plot for Maximum Line Loading
    fig_max, ax_max = plt.subplots(1, 1, subplot_kw={"projection": cartopy.crs.PlateCarree()})
    fig_max.set_size_inches(15, 17)
    line_colors_max = [cmap(value) for value in max_loading]  # Direct use of calculated max_loading as colors
    network.plot(ax=ax_max, bus_sizes=0, line_colors=line_colors_max, title="Maximum Line Loading")
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1)), ax=ax_max, orientation='vertical', fraction=0.02, pad=0.02, label='Line Loading as % of Nominal Capacity')

    # Adding map features
    ax_max.add_feature(cartopy.feature.LAND)
    ax_max.add_feature(cartopy.feature.COASTLINE)
    ax_max.add_feature(cartopy.feature.BORDERS, linestyle=':')
    ax_max.add_feature(cartopy.feature.OCEAN)
    ax_max.set_extent([-7, 2, 49, 59], crs=cartopy.crs.PlateCarree())
    plt.tight_layout()
    plt.show()

    # Plot for Average Line Loading
    fig_avg, ax_avg = plt.subplots(1, 1, subplot_kw={"projection": cartopy.crs.PlateCarree()})
    fig_avg.set_size_inches(8, 9)
    line_colors_avg = [cmap(value) for value in avg_loading_per_line]  # Direct use of calculated avg_loading_per_line as colors
    network.plot(ax=ax_avg, bus_sizes=0, line_colors=line_colors_avg, title="Average Line Loading")
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1)), ax=ax_avg, orientation='vertical', fraction=0.02, pad=0.02, label='Line Loading as % of Nominal Capacity')

    # Adding map features
    ax_avg.add_feature(cartopy.feature.LAND)
    ax_avg.add_feature(cartopy.feature.COASTLINE)
    ax_avg.add_feature(cartopy.feature.BORDERS, linestyle=':')
    ax_avg.add_feature(cartopy.feature.OCEAN)
    ax_avg.set_extent([-7, 2, 49, 59], crs=cartopy.crs.PlateCarree())
    plt.tight_layout()
    plt.show()

    # Print the average loading for the entire network
    print(f"Average line loading for the entire network: {avg_loading_network * 100:.2f}%")


def create_new_distribution(network, storage_type):
    tot_cap = calculate_specific_storage_capacity(network, storage_type)
    battery_updates = {
        "Beauly Battery": tot_cap * 0.01,
        "Peterhead Battery": tot_cap * 0.01,
        "Errochty Battery": tot_cap * 0.03,
        "Denny/Bonnybridge Battery": tot_cap * 0.07,
        "Neilston Battery": tot_cap * 0.04,
        "Strathaven Battery": tot_cap * 0.03,
        "Torness Battery": tot_cap * 0.02,
        "Eccles Battery": tot_cap * 0.02,
        "Harker Battery": tot_cap * 0.02,
        "Stella West Battery": tot_cap * 0.03,
        "Penwortham Battery": tot_cap * 0.05,
        "Deeside Battery": tot_cap * 0.05,
        "Daines Battery": tot_cap * 0.05,
        "Th. Marsh/Stocksbridge Battery": tot_cap * 0.05,
        "Thornton/Drax/Eggborough Battery": tot_cap * 0.03,
        "Keadby Battery": tot_cap * 0.02,
        "Ratcliffe Battery": tot_cap * 0.02,
        "Feckenham Battery": tot_cap * 0.05,
        "Walpole Battery": tot_cap * 0.02,
        "Bramford Battery": tot_cap * 0.02,
        "Pelham Battery": tot_cap * 0.03,
        "Sundon/East Claydon Battery": tot_cap * 0.03,
        "Melksham Battery": tot_cap * 0.04,
        "Bramley Battery": tot_cap * 0.02,
        "London Battery": tot_cap * 0.12,
        "Kemsley Battery": tot_cap * 0.03,
        "Sellindge Battery": tot_cap * 0.02,
        "Lovedean Battery": tot_cap * 0.03,
        "S.W.Penisula Battery": tot_cap * 0.04
    }

    return battery_updates

if __name__ == "__main__":
    network = pypsa.Network()
    network.import_from_csv_folder('LOPF_data')

    # # Prepare the network by adjusting line capacities, etc.
    # contingency_factor = 4
    # network.lines.s_max_pu *= contingency_factor

    # Initial optimization
    run_optimization(network)

    # set storage type to be optimised
    storage_type = "Battery"

    # Return values and create plots
    plot_storage_technology_capacity(network, storage_type)
    plot_curtailment(network)
    calculate_tot_marginal_price(network)
    plot_line_loading(network)

    battery_updates = create_new_distribution(network, storage_type)

    # Update storage units
    update_storage_units(network, battery_updates)

    # Find total storage amount
    print(f"Global storage limit: {calculate_specific_storage_capacity(network, storage_type)}")

    # Optimise with new storage
    run_optimization(network)

    # Return values and create plots
    plot_storage_technology_capacity(network, storage_type)
    plot_curtailment(network)
    calculate_tot_marginal_price(network)
    plot_line_loading(network)





