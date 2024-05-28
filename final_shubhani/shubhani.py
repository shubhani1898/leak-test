import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

logo_image = "C:/Users/v-kgajjana/Desktop/final_shubhani/tasi.png"  # Change to your logo image path


# --- Data Generation Function ---
def generate_pressure_decay_data(part_id, leak_type, leak_rate, test_parameters):
    target_pressure = test_parameters["target_pressure"]
    fill_rate = test_parameters["fill_rate"]
    stab_duration = test_parameters["stab_duration"]
    test_duration = test_parameters["test_duration"]
    noise_std = 0.05

    prefill_time = 2
    prefill_pressure = np.linspace(0, 0.1 * target_pressure, prefill_time * 10)

    fill_time = int(target_pressure / fill_rate)
    fill_pressure = np.linspace(0.1 * target_pressure, target_pressure, fill_time * 10)

    stab_pressure = [target_pressure] * (stab_duration * 10)

    if leak_type == 'Defect_1':
        prefill_pressure -= leak_rate * np.arange(len(prefill_pressure)) / 10
        fill_pressure -= leak_rate * np.arange(len(fill_pressure)) / 10
        test_time = np.linspace(0, test_duration, test_duration * 10)
        test_pressure = target_pressure - leak_rate * test_time
    elif leak_type == 'Defect_2':
        test_time = np.linspace(0, test_duration, test_duration * 10)
        test_pressure = target_pressure - leak_rate * test_time
        noise = np.random.normal(0, noise_std, size=test_duration * 10)
        test_pressure = np.maximum(test_pressure + noise, 0)
    else:  # 'no_leak'
        test_time = np.linspace(0, test_duration, test_duration * 10)
        test_pressure = [target_pressure] * (test_duration * 10)

    time = np.concatenate([
        np.arange(len(prefill_pressure)) / 10,
        np.arange(len(fill_pressure)) / 10 + prefill_time,
        np.arange(len(stab_pressure)) / 10 + prefill_time + fill_time,
        test_time
    ])
    pressure = np.concatenate([prefill_pressure, fill_pressure, stab_pressure, test_pressure])
    phase = (['prefill'] * len(prefill_pressure) +
             ['fill'] * len(fill_pressure) +
             ['stab'] * len(stab_pressure) +
             ['test'] * len(test_pressure))
    label = leak_type

    data = pd.DataFrame({
        'Timestamp': time,
        'Pressure': pressure,
        'Phase': phase,
        'Part_ID': part_id,
        'Leak_Rate': leak_rate,
        'Target_Pressure': target_pressure,
        'Fill_Rate': fill_rate,
        'Stab_Duration': stab_duration,
        'Test_Duration': test_duration,
        'Leak_Type': label  # Note: This is for sample data, not for user input
    })
    return data


# --- Sample Data Creation ---
part_ids = ['Part1', 'Part2', 'Part3']  
leak_types = ['Defect_1', 'no_leak', 'Defect_2'] 

leak_rates_dict = {
    'Defect_1': 0.05, 
    'no_leak': 0.0,
    'Defect_2': 0.1
}

test_parameters = {
    'target_pressure': 5,
    'fill_rate': 0.5,
    'stab_duration': 5,
    'test_duration': 30
}

all_data = pd.DataFrame()
for part_id, leak_type in zip(part_ids, leak_types):
    leak_rate = leak_rates_dict[leak_type]
    data = generate_pressure_decay_data(part_id, leak_type, leak_rate, test_parameters)
    all_data = pd.concat([all_data, data], ignore_index=True)
    # ... (same function as before to generate pressure decay data)

# --- Sample Data Creation ---
# ... (same sample data creation as before)

# --- Streamlit App ---
model = joblib.load('svm_model.joblib')
scaler = joblib.load('scaler.joblib')
features = ['Timestamp', 'Pressure', 'Phase', 'Part_ID', 'Leak_Rate', 'Target_Pressure', 'Fill_Rate', 'Stab_Duration', 'Test_Duration']

st.title('Cts Leak Data  Predction')
st.image(logo_image, use_column_width=True) 

data_source = st.radio("Select Data Source:", ("Enter Data Manually", "Use Sample Data"))

if data_source == "Enter Data Manually":
    num_rows = st.number_input("Number of Data Points:", min_value=1, value=10)
    data = []

    for i in range(num_rows):
        with st.expander(f"Data Point {i+1}"):
            col1, col2, col3 = st.columns(3)

            timestamp = col1.number_input(f"Timestamp {i+1}", key=f"timestamp_{i}")
            pressure = col2.number_input(f"Pressure {i+1}", key=f"pressure_{i}")
            phase = col3.selectbox(f"Phase {i+1}", options=['prefill', 'fill', 'stab', 'test'], key=f"phase_{i}")
            part_id = st.text_input(f"Part ID {i+1}", value="Part1", key=f"part_id_{i}")

            leak_rate = st.number_input(f"Leak Rate {i+1}", key=f"leak_rate_{i}")
            target_pressure = st.number_input(f"Target Pressure {i+1}", key=f"target_pressure_{i}")
            fill_rate = st.number_input(f"Fill Rate {i+1}", key=f"fill_rate_{i}")
            stab_duration = st.number_input(f"Stabilization Duration {i+1}", key=f"stab_duration_{i}")
            test_duration = st.number_input(f"Test Duration {i+1}", key=f"test_duration_{i}")

            new_row = {'Timestamp': timestamp, 'Pressure': pressure, 'Phase': phase,
                       'Part_ID': part_id, 'Leak_Rate': leak_rate, 'Target_Pressure': target_pressure,
                       'Fill_Rate': fill_rate, 'Stab_Duration': stab_duration, 'Test_Duration': test_duration}
            data.append(new_row) 

    if st.button('Predict'):
        data = pd.DataFrame(data)
        data['Leak_Type'] = -1  # Placeholder
        data['Phase'] = data['Phase'].astype('category').cat.codes
        data['Part_ID'] = data['Part_ID'].astype('category').cat.codes
        X = data[features]
        X_scaled = scaler.transform(X)

        predictions = model.predict(X_scaled)

        leak_type_map = {
            0: "No Leak",
            1: "Defect_1",
            2: "Defect_2",
        }

        prediction_labels = [leak_type_map[pred] for pred in predictions]
        data['Predicted Leak Type'] = prediction_labels

        st.write("### Predictions:")
        st.dataframe(data[['Part_ID', 'Pressure', 'Timestamp', 'Predicted Leak Type']])  

        # Plot graphs
        st.write("### Pressure vs. Time:")
        fig_pressure, ax_pressure = plt.subplots()
        for part_id in data['Part_ID'].unique():
            part_data = data[data['Part_ID'] == part_id]
            ax_pressure.plot(part_data['Timestamp'], part_data['Pressure'], label=f'Part {part_id}')
        ax_pressure.set_xlabel('Time')
        ax_pressure.set_ylabel('Pressure')
        ax_pressure.legend()
        st.pyplot(fig_pressure)

        st.write("### Leak Type Distribution:")
        fig_leak_type, ax_leak_type = plt.subplots()
        leak_type_counts = data['Predicted Leak Type'].value_counts()
        ax_leak_type.bar(leak_type_counts.index, leak_type_counts.values)
        ax_leak_type.set_xlabel('Leak Type')
        ax_leak_type.set_ylabel('Count')
        st.pyplot(fig_leak_type)
else:  # Use sample data
    data = all_data
    
    # Preprocessing 
    data['Leak_Type'] = data['Leak_Type'].astype('category').cat.codes
    data['Phase'] = data['Phase'].astype('category').cat.codes
    data['Part_ID'] = data['Part_ID'].astype('category').cat.codes
    X = data[features]
    X_scaled = scaler.transform(X)
    
    # Predict leak types
    predictions = model.predict(X_scaled)

    # Display predictions
    leak_type_map = {
        0: "No Leak",
        1: "Defect_1",
        2: "Defect_2",
    }

    prediction_labels = [leak_type_map[pred] for pred in predictions]
    data['Predicted Leak Type'] = prediction_labels

    st.write("### Predictions:")
    st.dataframe(data[['Part_ID', 'Pressure', 'Timestamp', 'Predicted Leak Type']])  

    # Plot graphs
    st.write("### Pressure vs. Time:")
    fig_pressure, ax_pressure = plt.subplots()
    for part_id in data['Part_ID'].unique():
        part_data = data[data['Part_ID'] == part_id]
        ax_pressure.plot(part_data['Timestamp'], part_data['Pressure'], label=f'Part {part_id}')
    ax_pressure.set_xlabel('Time')
    ax_pressure.set_ylabel('Pressure')
    ax_pressure.legend()
    st.pyplot(fig_pressure)

    st.write("### Leak Type Distribution:")
    fig_leak_type, ax_leak_type = plt.subplots()
    leak_type_counts = data['Predicted Leak Type'].value_counts()
    ax_leak_type.bar(leak_type_counts.index, leak_type_counts.values)
    ax_leak_type.set_xlabel('Leak Type')
    ax_leak_type.set_ylabel('Count')
    st.pyplot(fig_leak_type)