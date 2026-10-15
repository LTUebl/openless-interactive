"""
OpenLESS - Automated Landing Error Scoring System Pipeline
Paper: Turner et al., 2024 (https://doi.org/10.1101/2024.11.28.24318160)
Adapted for OpenCap jump landing data in OpenCapData_71.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from scipy.signal import butter, filtfilt, argrelmin

# Full column headers
column_headers = [
    "Trial_ID",
    "Total_Score",
    "1_Knee_Flexion_IC_L",
    "1_Knee_Flexion_IC_R",
    "1_Knee_Flexion_IC_T",
    "2_Hip_Flexion_IC_L",
    "2_Hip_Flexion_IC_R",
    "2_Hip_Flexion_IC_T",
    "3_Trunk_Flexion_IC_T",
    "4_Ankle_Plantar_Flexion_IC_L",
    "4_Ankle_Plantar_Flexion_IC_R",
    "4_Ankle_Plantar_Flexion_IC_T",
    "5_Asymmetrical_Timing",
    "6_Asymmetrical_HeelToe_ToeHeel",
    "7_Lateral_Trunk_Flexion_IC_L",
    "7_Lateral_Trunk_Flexion_IC_R",
    "7_Lateral_Trunk_Flexion_IC_T",
    "8_Medial_Knee_IC_L",
    "8_Medial_Knee_IC_R",
    "8_Medial_Knee_IC_T",
    "9_Stance_Wide_IC_L",
    "9_Stance_Wide_IC_R",
    "9_Stance_Wide_IC_T",
    "10_Stance_Narrow_IC_L",
    "10_Stance_Narrow_IC_R",
    "10_Stance_Narrow_IC_T",
    "11_IR_Foot_IC_L",
    "11_IR_Foot_IC_R",
    "11_IR_Foot_IC_T",
    "12_ER_Foot_IC_L",
    "12_ER_Foot_IC_R",
    "12_ER_Foot_IC_T",
    "13_Knee_Flexion_Displacement_L",
    "13_Knee_Flexion_Displacement_R",
    "13_Knee_Flexion_Displacement_T",
    "14_Hip_Flexion_Displacement_L",
    "14_Hip_Flexion_Displacement_R",
    "14_Hip_Flexion_Displacement_T",
    "15_Trunk_Flexion_Max_T",
    "16_Medial_Knee_Max_L",
    "16_Medial_Knee_Max_R",
    "16_Medial_Knee_Max_T",
    "17_Asymmetrical_Load_T",
    "18_Joint_Displacement_T",
    "19_Overall_Impression_T",
    "1_Knee_Flexion_IC_L_deg",
    "1_Knee_Flexion_IC_R_deg",
    "2_Hip_Flexion_IC_L_deg",
    "2_Hip_Flexion_IC_R_deg",
    "3_Trunk_Flexion_IC_deg",
    "7_Lateral_Trunk_Flexion_IC_deg",
    "8_Medial_Knee_IC_L_deg",
    "8_Medial_Knee_IC_R_deg",
    "9_Stance_Hip_abd_IC_L_deg",
    "9_Stance_Hip_abd_IC_R_deg",
    "11_IR_Foot_IC_L_deg",
    "11_IR_Foot_IC_R_deg",
    "12_ER_Foot_IC_L_deg",
    "12_ER_Foot_IC_R_deg",
    "13_Knee_Flexion_Displacement_L_deg",
    "13_Knee_Flexion_Displacement_R_deg",
    "14_Hip_Flexion_Displacement_L_deg",
    "14_Hip_Flexion_Displacement_R_deg",
    "15_Trunk_Flexion_Max_deg",
    "16_Medial_Knee_Max_L_deg",
    "16_Medial_Knee_Max_R_deg",
    "17_Asymmetrical_Load_L_deg",
    "17_Asymmetrical_Load_R_deg",
]

column_headers_short = [
    "Trial_ID",
    "Total_Score",
    "1_Knee_Flexion_IC_T",
    "2_Hip_Flexion_IC_T",
    "3_Trunk_Flexion_IC_T",
    "4_Ankle_Plantar_Flexion_IC_T",
    "5_Asymmetrical_Timing",
    "6_Asymmetrical_HeelToe_ToeHeel",
    "7_Lateral_Trunk_Flexion_IC_T",
    "8_Medial_Knee_IC_T",
    "9_Stance_Wide_IC_T",
    "10_Stance_Narrow_IC_T",
    "11_IR_Foot_IC_T",
    "12_ER_Foot_IC_T",
    "13_Knee_Flexion_Displacement_T",
    "14_Hip_Flexion_Displacement_T",
    "15_Trunk_Flexion_Max_T",
    "16_Medial_Knee_Max_T",
    "17_Asymmetrical_Load_T",
    "18_Joint_Displacement_T",
    "19_Overall_Impression_T",
]

def load_coord_file(file_name):
    data = np.loadtxt(file_name, skiprows=6)
    time = data[:, 1]
    y_coords_left_toes = data[:, 45]
    y_coords_left_heel = data[:, 51]
    y_coords_right_toes = data[:, 54]
    y_coords_right_heel = data[:, 60]
    return (
        data,
        time,
        y_coords_left_toes,
        y_coords_left_heel,
        y_coords_right_toes,
        y_coords_right_heel,
    )

def apply_butterworth_filter_coords(
    y_coords_left_toes,
    y_coords_left_heel,
    y_coords_right_toes,
    y_coords_right_heel,
    sampling_freq,
    coord_cutoff_freq,
):
    cutoff_freq = coord_cutoff_freq  # Hz
    fs = sampling_freq  # Sampling frequency
    nyquist_freq = fs / 2  # Nyquist frequency
    filter_order = 4  # Filter order

    b, a = butter(filter_order, cutoff_freq / nyquist_freq, btype="low")
    left_toes_filtered = filtfilt(b, a, y_coords_left_toes)
    left_heel_filtered = filtfilt(b, a, y_coords_left_heel)
    right_toes_filtered = filtfilt(b, a, y_coords_right_toes)
    right_heel_filtered = filtfilt(b, a, y_coords_right_heel)

    return (
        left_toes_filtered,
        left_heel_filtered,
        right_toes_filtered,
        right_heel_filtered,
    )

def apply_butterworth_filter(file_path, sampling_freq, cutoff_freq):
    if isinstance(file_path, pd.DataFrame):
        data = file_path
    else:
        data = pd.read_csv(file_path)

    filtered_data = data.copy()
    for column in data.columns:
        if column != "time":
            column_data = data[column].values
            fs = sampling_freq
            nyquist_freq = fs / 2
            filter_order = 4
            b, a = butter(filter_order, cutoff_freq / nyquist_freq, btype="low")
            padlen = len(column_data) // 2
            filtered_column_data = filtfilt(b, a, column_data, padlen=padlen)
            filtered_data[column] = filtered_column_data
    return filtered_data

def id_initialcontact_toes(toes_filtered, time):
    toes_local_minima_indices = argrelmin(toes_filtered)
    start_threshold = toes_filtered[0] - 0.2
    first_minima_index = None
    for index in toes_local_minima_indices[0]:
        if toes_filtered[index] < start_threshold:
            first_minima_index = index
            break
    toes_first_minima_time = (
        time[first_minima_index] if first_minima_index is not None else None
    )
    toes_first_minima_y_coord = (
        toes_filtered[first_minima_index] if first_minima_index is not None else None
    )
    return toes_first_minima_time, toes_first_minima_y_coord, first_minima_index

def id_initialcontact_heels(heel_filtered, time):
    heel_local_minima_indices = argrelmin(heel_filtered)
    start_threshold = heel_filtered[0] - 0.2
    first_minima_index = None
    for index in heel_local_minima_indices[0]:
        if heel_filtered[index] < start_threshold:
            first_minima_index = index
            break
    heel_first_minima_time = (
        time[first_minima_index] if first_minima_index is not None else None
    )
    heel_first_minima_y_coord = (
        heel_filtered[first_minima_index] if first_minima_index is not None else None
    )
    return heel_first_minima_time, first_minima_index

def id_takeoff_toes(toes_filtered, time, first_minima_index):
    toes_local_minima_indices = argrelmin(toes_filtered)
    toes_next_peak_index = (
        np.argmax(toes_filtered[first_minima_index:]) + first_minima_index
    )
    toes_next_peak_time = time[toes_next_peak_index]
    toes_next_peak_value = toes_filtered[toes_next_peak_index]
    last_minima_index = toes_local_minima_indices[0][
        np.where(toes_local_minima_indices[0] < toes_next_peak_index)[0][-1]
    ]
    toes_second_minima_time = (
        time[last_minima_index] if last_minima_index is not None else None
    )
    toes_second_minima_y_coord = (
        toes_filtered[last_minima_index] if last_minima_index is not None else None
    )
    toe_off_index = last_minima_index

    return toe_off_index, toes_second_minima_time

def compare_initialcontact_toe_vs_heel(
    toes_first_minima_time,
    toe_first_minima_index,
    heel_first_minima_time,
    heel_first_minima_index,
    time,
):
    if toes_first_minima_time < heel_first_minima_time:
        y_min_index = toe_first_minima_index
    else:
        y_min_index = heel_first_minima_index
    y_min_index_time = time[y_min_index]
    return y_min_index, y_min_index_time

def plot_filtered_data(
    time,
    heel_filtered,
    toes_filtered,
    y_min_index_time,
    toes_second_minima_time,
    title,
    side,
    save_path=None,
):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, toes_filtered, label="Toes", color="blue")
    ax.plot(time, heel_filtered, label="Heel", color="orange")
    ax.axvline(y_min_index_time, color="green", linestyle="--", label="Initial Contact")
    ax.axvline(toes_second_minima_time, color="red", linestyle="--", label="Toe-off")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("y-coordinate (m)")
    ax.legend()
    plt.title(f"{title} - {side.capitalize()} Leg Quality Check")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.show()

def generate_plot_table_quality_check(
    time,
    heel_filtered,
    toes_filtered,
    y_min_index_time,
    toes_second_minima_time,
    title,
    side,
    output_dir=None,
):
    save_path = os.path.join(output_dir, f"{title}_{side}_qc.png") if output_dir else None
    plot_filtered_data(
        time,
        heel_filtered,
        toes_filtered,
        y_min_index_time,
        toes_second_minima_time,
        title,
        side,
        save_path=save_path,
    )

    table = PrettyTable()
    table.field_names = ["Event", "Time (s)"]
    table.add_row(["time of initial contact", "{:.3f}".format(y_min_index_time)])
    table.add_row(["toe-off", "{:.3f}".format(toes_second_minima_time)])
    table.add_row(
        [
            f"{side} stance duration",
            "{:.3f}".format(toes_second_minima_time - y_min_index_time),
        ]
    )
    table.title = f"Results for {title} {side}"
    print(table)

def apply_stance_phase_to_motion_file(
    data,
    time,
    y_min_index,
    toe_off_index,
    kinematics_name,
    start_column: int,
    end_column: int,
):
    stance_coords = pd.DataFrame(data)
    stance_coords = stance_coords[
        (stance_coords.iloc[:, 1] >= time[y_min_index])
        & (stance_coords.iloc[:, 1] <= time[toe_off_index])
    ]
    stance_coords.reset_index(drop=True, inplace=True)

    with open(kinematics_name, "r") as f:
        header_row = 0
        for idx, line in enumerate(f):
            if line.startswith("endheader"):
                header_row = idx + 1
                break
    df_kinematics = pd.read_csv(kinematics_name, sep="\t", skiprows=header_row)
    stance_kinematics = pd.DataFrame(df_kinematics)
    cols = (
        list(range(0, 7))
        + list(range(start_column, end_column + 1))
        + list(range(21, 25))
    )
    kinematics = stance_kinematics.iloc[:, cols].copy()
    kinematics = kinematics[
        (kinematics.iloc[:, 0] >= time[y_min_index])
        & (kinematics.iloc[:, 0] <= time[toe_off_index])
    ]
    kinematics.reset_index(drop=True, inplace=True)
    return kinematics, stance_coords

def save_filtered_data(filtered_data, title, side_of_body: str, output_dir="."):
    filtered_file_name = f"{side_of_body}_kin_{title}_filtered.csv"
    file_path = os.path.join(output_dir, filtered_file_name)
    filtered_data.to_csv(file_path, index=False)
    print(f"filtered data saved to: {file_path}")
    return file_path

# ==================== LESS Scoring Functions ====================

def score_knee_flexion(
    ic_error,
    max_error,
    Trial_ID,
    ic_knee_flexion_deg_r,
    max_knee_flexion_deg_r,
    ic_knee_flexion_deg_l,
    max_knee_flexion_deg_l,
    LESS_results,
    LESS_results_short,
):
    ic_knee_flexion_r = 1 if ic_knee_flexion_deg_r < ic_error else 0
    max_knee_flexion_r = 1 if max_knee_flexion_deg_r < max_error else 0
    ic_knee_flexion_l = 1 if ic_knee_flexion_deg_l < ic_error else 0
    max_knee_flexion_l = 1 if max_knee_flexion_deg_l < max_error else 0
    ic_knee_flexion_total = 1 if ic_knee_flexion_r or ic_knee_flexion_l else 0
    max_knee_flexion_total = 1 if max_knee_flexion_r or max_knee_flexion_l else 0

    LESS_results.at[Trial_ID, "1_Knee_Flexion_IC_R"] = ic_knee_flexion_r
    LESS_results.at[Trial_ID, "13_Knee_Flexion_Displacement_R"] = max_knee_flexion_r
    LESS_results.at[Trial_ID, "1_Knee_Flexion_IC_L"] = ic_knee_flexion_l
    LESS_results.at[Trial_ID, "13_Knee_Flexion_Displacement_L"] = max_knee_flexion_l

    LESS_results.at[Trial_ID, "1_Knee_Flexion_IC_T"] = ic_knee_flexion_total
    LESS_results.at[Trial_ID, "13_Knee_Flexion_Displacement_T"] = max_knee_flexion_total
    LESS_results_short.at[Trial_ID, "1_Knee_Flexion_IC_T"] = ic_knee_flexion_total
    LESS_results_short.at[Trial_ID, "13_Knee_Flexion_Displacement_T"] = max_knee_flexion_total

    return ic_knee_flexion_total, max_knee_flexion_total

def id_knee_flexion_error(Trial_ID, input_df_r, input_df_l, LESS_results, LESS_results_short):
    ic_error = 30.0
    max_error = 65.0

    ic_knee_flexion_deg_r = input_df_r.loc[0, "knee_angle_r"]
    max_knee_flexion_deg_r = input_df_r.loc[
        0 : input_df_r["knee_angle_r"].idxmax(), "knee_angle_r"
    ].max()
    ic_knee_flexion_deg_l = input_df_l.loc[0, "knee_angle_l"]
    max_knee_flexion_deg_l = input_df_l.loc[
        0 : input_df_l["knee_angle_l"].idxmax(), "knee_angle_l"
    ].max()

    ic_knee_flexion_total, max_knee_flexion_total = score_knee_flexion(
        ic_error,
        max_error,
        Trial_ID,
        ic_knee_flexion_deg_r,
        max_knee_flexion_deg_r,
        ic_knee_flexion_deg_l,
        max_knee_flexion_deg_l,
        LESS_results,
        LESS_results_short,
    )

    LESS_results.at[Trial_ID, "1_Knee_Flexion_IC_R_deg"] = ic_knee_flexion_deg_r
    LESS_results.at[Trial_ID, "13_Knee_Flexion_Displacement_R_deg"] = max_knee_flexion_deg_r
    LESS_results.at[Trial_ID, "1_Knee_Flexion_IC_L_deg"] = ic_knee_flexion_deg_l
    LESS_results.at[Trial_ID, "13_Knee_Flexion_Displacement_L_deg"] = max_knee_flexion_deg_l

    return (
        max_knee_flexion_deg_r,
        max_knee_flexion_deg_l,
        ic_knee_flexion_total,
        max_knee_flexion_total,
    )

def score_hip_flexion(
    ic_error,
    max_error,
    Trial_ID,
    ic_hip_flexion_deg_r,
    max_hip_flexion_deg_r,
    ic_hip_flexion_deg_l,
    max_hip_flexion_deg_l,
    LESS_results,
    LESS_results_short,
):
    ic_hip_flexion_r = 1 if ic_hip_flexion_deg_r < ic_error else 0
    max_hip_flexion_r = 1 if max_hip_flexion_deg_r < max_error else 0
    ic_hip_flexion_l = 1 if ic_hip_flexion_deg_l < ic_error else 0
    max_hip_flexion_l = 1 if max_hip_flexion_deg_l < max_error else 0
    ic_hip_flexion_total = 1 if ic_hip_flexion_r or ic_hip_flexion_l else 0
    max_hip_flexion_total = 1 if max_hip_flexion_r or max_hip_flexion_l else 0

    LESS_results.at[Trial_ID, "2_Hip_Flexion_IC_R"] = ic_hip_flexion_r
    LESS_results.at[Trial_ID, "14_Hip_Flexion_Displacement_R"] = max_hip_flexion_r
    LESS_results.at[Trial_ID, "2_Hip_Flexion_IC_L"] = ic_hip_flexion_l
    LESS_results.at[Trial_ID, "14_Hip_Flexion_Displacement_L"] = max_hip_flexion_l
    LESS_results.at[Trial_ID, "2_Hip_Flexion_IC_T"] = ic_hip_flexion_total
    LESS_results.at[Trial_ID, "14_Hip_Flexion_Displacement_T"] = max_hip_flexion_total
    LESS_results_short.at[Trial_ID, "2_Hip_Flexion_IC_T"] = ic_hip_flexion_total
    LESS_results_short.at[Trial_ID, "14_Hip_Flexion_Displacement_T"] = max_hip_flexion_total
    return ic_hip_flexion_total, max_hip_flexion_total

def id_hip_flexion_error(Trial_ID, input_df_r, input_df_l, LESS_results, LESS_results_short):
    ic_error = 30.0
    max_error = 45.0

    ic_hip_flexion_deg_r = input_df_r.loc[0, "hip_flexion_r"]
    max_knee_angle_row_r = input_df_r.loc[input_df_r["knee_angle_r"].idxmax()]
    max_hip_flexion_deg_r = max_knee_angle_row_r["hip_flexion_r"]
    ic_hip_flexion_deg_l = input_df_l.loc[0, "hip_flexion_l"]
    max_knee_angle_row_l = input_df_l.loc[input_df_l["knee_angle_l"].idxmax()]
    max_hip_flexion_deg_l = max_knee_angle_row_l["hip_flexion_l"]

    ic_hip_flexion_total, max_hip_flexion_total = score_hip_flexion(
        ic_error,
        max_error,
        Trial_ID,
        ic_hip_flexion_deg_r,
        max_hip_flexion_deg_r,
        ic_hip_flexion_deg_l,
        max_hip_flexion_deg_l,
        LESS_results,
        LESS_results_short,
    )

    LESS_results.at[Trial_ID, "2_Hip_Flexion_IC_R_deg"] = ic_hip_flexion_deg_r
    LESS_results.at[Trial_ID, "14_Hip_Flexion_Displacement_R_deg"] = max_hip_flexion_deg_r
    LESS_results.at[Trial_ID, "2_Hip_Flexion_IC_L_deg"] = ic_hip_flexion_deg_l
    LESS_results.at[Trial_ID, "14_Hip_Flexion_Displacement_L_deg"] = max_hip_flexion_deg_l

    return (
        max_hip_flexion_deg_r,
        max_hip_flexion_deg_l,
        ic_hip_flexion_total,
        max_hip_flexion_total,
    )

def score_trunk_flexion(ic_error, max_error, Trial_ID, ic_trunk_flexion_deg, max_trunk_flexion_deg, LESS_results, LESS_results_short):
    ic_trunk_flexion_total = 1 if ic_trunk_flexion_deg > ic_error else 0
    max_trunk_flexion_total = (
        1 if (ic_trunk_flexion_deg - max_trunk_flexion_deg) < max_error else 0
    )

    LESS_results.at[Trial_ID, "3_Trunk_Flexion_IC_T"] = ic_trunk_flexion_total
    LESS_results.at[Trial_ID, "15_Trunk_Flexion_Max_T"] = max_trunk_flexion_total
    LESS_results_short.at[Trial_ID, "3_Trunk_Flexion_IC_T"] = ic_trunk_flexion_total
    LESS_results_short.at[Trial_ID, "15_Trunk_Flexion_Max_T"] = max_trunk_flexion_total
    return ic_trunk_flexion_total, max_trunk_flexion_total

def id_trunk_flexion_error(Trial_ID, input_df_r, LESS_results, LESS_results_short):
    ic_error = -10.0
    max_error = 10.0

    ic_trunk_flexion_deg = input_df_r.loc[0, "lumbar_extension"]
    max_knee_angle_row = input_df_r.loc[input_df_r["knee_angle_r"].idxmax()]
    max_trunk_flexion_deg = max_knee_angle_row["lumbar_extension"]

    ic_trunk_flexion_total, max_trunk_flexion_total = score_trunk_flexion(
        ic_error, max_error, Trial_ID, ic_trunk_flexion_deg, max_trunk_flexion_deg, LESS_results, LESS_results_short
    )

    LESS_results.at[Trial_ID, "3_Trunk_Flexion_IC_deg"] = ic_trunk_flexion_deg
    LESS_results.at[Trial_ID, "15_Trunk_Flexion_Max_deg"] = max_trunk_flexion_deg

    return (ic_trunk_flexion_total, max_trunk_flexion_total)

def id_ankle_plantarflexion_error(Trial_ID, coord_file, sampling_freq, coord_cutoff_freq, LESS_results, LESS_results_short):
    (
        data,
        time,
        y_coords_left_toes,
        y_coords_left_heel,
        y_coords_right_toes,
        y_coords_right_heel,
    ) = load_coord_file(coord_file)

    left_toes_filtered, left_heel_filtered, right_toes_filtered, right_heel_filtered = (
        apply_butterworth_filter_coords(
            y_coords_left_toes,
            y_coords_left_heel,
            y_coords_right_toes,
            y_coords_right_heel,
            sampling_freq,
            coord_cutoff_freq,
        )
    )

    toes_first_minima_time, toes_first_minima_y_coord, lt_first_minima_index = (
        id_initialcontact_toes(left_toes_filtered, time)
    )
    heel_first_minima_time, lh_first_minima_index = id_initialcontact_heels(
        left_heel_filtered, time
    )
    ankle_PF_left = 1 if (toes_first_minima_time) >= (heel_first_minima_time) else 0
    LESS_results.at[Trial_ID, "4_Ankle_Plantar_Flexion_IC_L"] = ankle_PF_left

    toes_first_minima_time_r, toes_first_minima_y_coord, rt_first_minima_index = (
        id_initialcontact_toes(right_toes_filtered, time)
    )
    heel_first_minima_time_r, rh_first_minima_index = id_initialcontact_heels(
        right_heel_filtered, time
    )
    ankle_PF_right = 1 if (toes_first_minima_time_r) >= (heel_first_minima_time_r) else 0
    LESS_results.at[Trial_ID, "4_Ankle_Plantar_Flexion_IC_R"] = ankle_PF_right

    ic_ankle_PF_total = 1 if ankle_PF_right or ankle_PF_left else 0
    LESS_results.at[Trial_ID, "4_Ankle_Plantar_Flexion_IC_T"] = ic_ankle_PF_total
    LESS_results_short.at[Trial_ID, "4_Ankle_Plantar_Flexion_IC_T"] = ic_ankle_PF_total
    return ankle_PF_left, ankle_PF_right, ic_ankle_PF_total

def id_asymmetric_timing_error(Trial_ID, input_df_r, input_df_l, LESS_results, LESS_results_short):
    timing_error = 0.034

    ic_timing_r = input_df_r.loc[0, "time"]
    ic_timing_l = input_df_l.loc[0, "time"]
    ic_timing_asymmetric_total = 1 if abs(ic_timing_r - ic_timing_l) > timing_error else 0
    LESS_results.at[Trial_ID, "5_Asymmetrical_Timing"] = ic_timing_asymmetric_total
    LESS_results_short.at[Trial_ID, "5_Asymmetrical_Timing"] = ic_timing_asymmetric_total
    return ic_timing_asymmetric_total

def asymmetric_landing_error(Trial_ID, ankle_PF_right, ankle_PF_left, LESS_results, LESS_results_short):
    if ankle_PF_right == ankle_PF_left:
        ic_landing_asymmetric_total = 0
    else:
        ic_landing_asymmetric_total = 1

    LESS_results.at[Trial_ID, "6_Asymmetrical_HeelToe_ToeHeel"] = ic_landing_asymmetric_total
    LESS_results_short.at[Trial_ID, "6_Asymmetrical_HeelToe_ToeHeel"] = ic_landing_asymmetric_total
    return ic_landing_asymmetric_total

def id_lateral_trunk_flexion_error(Trial_ID, input_df_r, LESS_results, LESS_results_short):
    trunk_lateral_error = 5.0
    ic_trunk_lateral_deg = input_df_r.loc[0, "lumbar_bending"]

    ic_trunk_lateral_r = 1 if ic_trunk_lateral_deg > trunk_lateral_error else 0
    ic_trunk_lateral_l = 1 if ic_trunk_lateral_deg < -(trunk_lateral_error) else 0
    ic_trunk_lateral_total = 1 if ic_trunk_lateral_r or ic_trunk_lateral_l else 0

    LESS_results.at[Trial_ID, "7_Lateral_Trunk_Flexion_IC_deg"] = ic_trunk_lateral_deg
    LESS_results.at[Trial_ID, "7_Lateral_Trunk_Flexion_IC_R"] = ic_trunk_lateral_r
    LESS_results.at[Trial_ID, "7_Lateral_Trunk_Flexion_IC_L"] = ic_trunk_lateral_l
    LESS_results.at[Trial_ID, "7_Lateral_Trunk_Flexion_IC_T"] = ic_trunk_lateral_total
    LESS_results_short.at[Trial_ID, "7_Lateral_Trunk_Flexion_IC_T"] = ic_trunk_lateral_total
    return ic_trunk_lateral_total

def knee_valgus_boolean(hip_rotation_deg, hip_adduction_deg):
    knee_valgus = (
        1
        if hip_rotation_deg > 8.0
        and hip_adduction_deg > -1
        or hip_adduction_deg > 1
        else 0
    )
    return knee_valgus

def id_knee_valgus_error(Trial_ID, input_df_r, input_df_l, LESS_results, LESS_results_short):
    ic_hip_rotation_deg_r = input_df_r.loc[0, "hip_rotation_r"]
    max_knee_angle_row_r = input_df_r.loc[input_df_r["knee_angle_r"].idxmax()]
    max_hip_rotation_deg_r = max_knee_angle_row_r["hip_rotation_r"]
    ic_hip_rotation_deg_l = input_df_l.loc[0, "hip_rotation_l"]
    max_knee_angle_row_l = input_df_l.loc[input_df_l["knee_angle_l"].idxmax()]
    max_hip_rotation_deg_l = max_knee_angle_row_l["hip_rotation_l"]

    ic_hip_adduction_deg_r = input_df_r.loc[0, "hip_adduction_r"]
    max_hip_adduction_deg_r = max_knee_angle_row_r["hip_adduction_r"]
    ic_hip_adduction_deg_l = input_df_l.loc[0, "hip_adduction_l"]
    max_hip_adduction_deg_l = max_knee_angle_row_l["hip_adduction_l"]

    ic_knee_valgus_r = knee_valgus_boolean(ic_hip_rotation_deg_r, ic_hip_adduction_deg_r)
    max_knee_valgus_r = knee_valgus_boolean(max_hip_rotation_deg_r, max_hip_adduction_deg_r)
    ic_knee_valgus_l = knee_valgus_boolean(ic_hip_rotation_deg_l, ic_hip_adduction_deg_l)
    max_knee_valgus_l = knee_valgus_boolean(max_hip_rotation_deg_l, max_hip_adduction_deg_l)

    ic_knee_valgus_total = 1 if ic_knee_valgus_r or ic_knee_valgus_l else 0
    max_knee_valgus_total = 1 if max_knee_valgus_r or max_knee_valgus_l else 0

    LESS_results.at[Trial_ID, "8_Medial_Knee_IC_R_deg"] = ic_hip_rotation_deg_r
    LESS_results.at[Trial_ID, "16_Medial_Knee_Max_R_deg"] = max_hip_rotation_deg_r
    LESS_results.at[Trial_ID, "8_Medial_Knee_IC_R"] = ic_knee_valgus_r
    LESS_results.at[Trial_ID, "16_Medial_Knee_Max_R"] = max_knee_valgus_r

    LESS_results.at[Trial_ID, "8_Medial_Knee_IC_L_deg"] = ic_hip_rotation_deg_l
    LESS_results.at[Trial_ID, "16_Medial_Knee_Max_L_deg"] = max_hip_rotation_deg_l
    LESS_results.at[Trial_ID, "8_Medial_Knee_IC_L"] = ic_knee_valgus_l
    LESS_results.at[Trial_ID, "16_Medial_Knee_Max_L"] = max_knee_valgus_l
    LESS_results.at[Trial_ID, "8_Medial_Knee_IC_T"] = ic_knee_valgus_total
    LESS_results.at[Trial_ID, "16_Medial_Knee_Max_T"] = max_knee_valgus_total
    LESS_results_short.at[Trial_ID, "8_Medial_Knee_IC_T"] = ic_knee_valgus_total
    LESS_results_short.at[Trial_ID, "16_Medial_Knee_Max_T"] = max_knee_valgus_total
    return ic_knee_valgus_total, max_knee_valgus_total

def id_wide_stance_error(Trial_ID, input_df_r, input_df_l, LESS_results, LESS_results_short):
    wide_stance_error = -12.0

    ic_hip_add_deg_r = input_df_r.loc[0, "hip_adduction_r"]
    ic_hip_add_deg_l = input_df_l.loc[0, "hip_adduction_l"]

    ic_hip_wide_r = 1 if ic_hip_add_deg_r < wide_stance_error else 0
    ic_hip_wide_l = 1 if ic_hip_add_deg_l < wide_stance_error else 0
    ic_hip_wide_total = 1 if ic_hip_wide_r or ic_hip_wide_l else 0

    LESS_results.at[Trial_ID, "9_Stance_Hip_abd_IC_R_deg"] = ic_hip_add_deg_r
    LESS_results.at[Trial_ID, "9_Stance_Wide_IC_R"] = ic_hip_wide_r
    LESS_results.at[Trial_ID, "9_Stance_Hip_abd_IC_L_deg"] = ic_hip_add_deg_l
    LESS_results.at[Trial_ID, "9_Stance_Wide_IC_L"] = ic_hip_wide_l
    LESS_results.at[Trial_ID, "9_Stance_Wide_IC_T"] = ic_hip_wide_total
    LESS_results_short.at[Trial_ID, "9_Stance_Wide_IC_T"] = ic_hip_wide_total
    return ic_hip_wide_total, ic_hip_add_deg_r, ic_hip_add_deg_l

def id_narrow_stance_error(Trial_ID, ic_hip_add_deg_r, ic_hip_add_deg_l, LESS_results, LESS_results_short):
    narrow_stance_error = 0

    ic_hip_narrow_r = 1 if ic_hip_add_deg_r > narrow_stance_error else 0
    ic_hip_narrow_l = 1 if ic_hip_add_deg_l > narrow_stance_error else 0
    ic_hip_narrow_total = 1 if ic_hip_narrow_r or ic_hip_narrow_l else 0
    LESS_results.at[Trial_ID, "10_Stance_Narrow_IC_R"] = ic_hip_narrow_r
    LESS_results.at[Trial_ID, "10_Stance_Narrow_IC_L"] = ic_hip_narrow_l
    LESS_results.at[Trial_ID, "10_Stance_Narrow_IC_T"] = ic_hip_narrow_total
    LESS_results_short.at[Trial_ID, "10_Stance_Narrow_IC_T"] = ic_hip_narrow_total
    return ic_hip_narrow_total

def id_foot_inward_error(Trial_ID, input_df_r, input_df_l, LESS_results, LESS_results_short):
    inward_error = 10.0

    ic_subtalar_deg_r = input_df_r.loc[0, "subtalar_angle_r"]
    ic_subtalar_deg_l = input_df_l.loc[0, "subtalar_angle_l"]
    ic_foot_inward_r = 1 if ic_subtalar_deg_r > inward_error else 0
    ic_foot_inward_l = 1 if ic_subtalar_deg_l > inward_error else 0
    ic_foot_inward_total = 1 if ic_foot_inward_r or ic_foot_inward_l else 0
    LESS_results.at[Trial_ID, "11_IR_Foot_IC_R"] = ic_foot_inward_r
    LESS_results.at[Trial_ID, "11_IR_Foot_IC_R_deg"] = ic_subtalar_deg_r
    LESS_results.at[Trial_ID, "11_IR_Foot_IC_L"] = ic_foot_inward_l
    LESS_results.at[Trial_ID, "11_IR_Foot_IC_L_deg"] = ic_subtalar_deg_l
    LESS_results.at[Trial_ID, "11_IR_Foot_IC_T"] = ic_foot_inward_total
    LESS_results_short.at[Trial_ID, "11_IR_Foot_IC_T"] = ic_foot_inward_total
    return ic_foot_inward_total

def id_foot_outward_error(Trial_ID, input_df_r, input_df_l, LESS_results, LESS_results_short):
    outward_error = -15.0

    ic_subtalar_deg_r = input_df_r.loc[0, "subtalar_angle_r"]
    ic_subtalar_deg_l = input_df_l.loc[0, "subtalar_angle_l"]
    ic_foot_outward_r = 1 if ic_subtalar_deg_r < outward_error else 0
    ic_foot_outward_l = 1 if ic_subtalar_deg_l < outward_error else 0
    ic_foot_outward_total = 1 if ic_foot_outward_r or ic_foot_outward_l else 0
    LESS_results.at[Trial_ID, "12_ER_Foot_IC_R"] = ic_foot_outward_r
    LESS_results.at[Trial_ID, "12_ER_Foot_IC_R_deg"] = ic_subtalar_deg_r
    LESS_results.at[Trial_ID, "12_ER_Foot_IC_L"] = ic_foot_outward_l
    LESS_results.at[Trial_ID, "12_ER_Foot_IC_L_deg"] = ic_subtalar_deg_l
    LESS_results.at[Trial_ID, "12_ER_Foot_IC_T"] = ic_foot_outward_total
    LESS_results_short.at[Trial_ID, "12_ER_Foot_IC_T"] = ic_foot_outward_total
    return ic_foot_outward_total

def id_asymmetric_loading_error(Trial_ID, input_df_r, input_df_l, LESS_results, LESS_results_short):
    loading_error = 5

    max_knee_angle_row_r = input_df_r.loc[input_df_r["knee_angle_r"].idxmax()]
    max_hip_abd_deg_r = max_knee_angle_row_r["hip_adduction_r"]
    max_knee_angle_row_l = input_df_l.loc[input_df_l["knee_angle_l"].idxmax()]
    max_hip_abd_deg_l = max_knee_angle_row_l["hip_adduction_l"]

    max_asymmetric_loading_total = (
        1 if abs(max_hip_abd_deg_r - max_hip_abd_deg_l) > loading_error else 0
    )

    LESS_results.at[Trial_ID, "17_Asymmetrical_Load_R_deg"] = max_hip_abd_deg_r
    LESS_results.at[Trial_ID, "17_Asymmetrical_Load_L_deg"] = max_hip_abd_deg_l
    LESS_results.at[Trial_ID, "17_Asymmetrical_Load_T"] = max_asymmetric_loading_total
    LESS_results_short.at[Trial_ID, "17_Asymmetrical_Load_T"] = max_asymmetric_loading_total
    return max_asymmetric_loading_total

def id_joint_displacement_error(
    Trial_ID,
    max_hip_flexion_total,
    max_trunk_flexion_total,
    max_knee_flexion_total,
    max_knee_flexion_deg_r,
    max_knee_flexion_deg_l,
    max_hip_flexion_deg_r,
    max_hip_flexion_deg_l,
    LESS_results,
    LESS_results_short,
):
    max_knee_error = 55
    max_hip_error = 73

    if (
        (max_hip_flexion_total == 0)
        and (max_trunk_flexion_total == 0)
        and (max_knee_flexion_total == 0)
    ):
        if (max_knee_flexion_deg_r >= max_knee_error) and (
            max_knee_flexion_deg_l >= max_knee_error
        ):
            if (max_hip_flexion_deg_r >= max_hip_error) and (
                max_hip_flexion_deg_l >= max_hip_error
            ):
                joint_displacement_total = 0
            else:
                joint_displacement_total = 1
        else:
            joint_displacement_total = 1
    else:
        joint_displacement_total = 2

    LESS_results.at[Trial_ID, "18_Joint_Displacement_T"] = joint_displacement_total
    LESS_results_short.at[Trial_ID, "18_Joint_Displacement_T"] = joint_displacement_total
    return joint_displacement_total

def id_overall_impression_error(
    Trial_ID,
    max_hip_flexion_total,
    max_trunk_flexion_total,
    max_knee_flexion_total,
    max_knee_valgus_total,
    ic_knee_valgus_total,
    LESS_results,
    LESS_results_short,
):
    if (
        max_hip_flexion_total == 0
        and max_trunk_flexion_total == 0
        and max_knee_flexion_total == 0
        and max_knee_valgus_total == 0
    ):
        overall_impression_total = 0
    elif (
        max_hip_flexion_total == 1
        or max_trunk_flexion_total == 1
        or max_knee_flexion_total == 1
    ) and (max_knee_valgus_total == 1 or ic_knee_valgus_total == 1):
        overall_impression_total = 2
    else:
        overall_impression_total = 1

    LESS_results.at[Trial_ID, "19_Overall_Impression_T"] = overall_impression_total
    LESS_results_short.at[Trial_ID, "19_Overall_Impression_T"] = overall_impression_total
    return overall_impression_total

def process_bilateral_LESS(
    right_trial_file,
    left_trial_file,
    coord_file,
    LESS_results,
    LESS_results_short,
    sampling_freq,
    coord_cutoff_freq,
):
    input_df_r = pd.read_csv(right_trial_file)
    input_df_l = pd.read_csv(left_trial_file)
    Trial_ID = (
        os.path.basename(right_trial_file)
        .replace("right_kin_", "")
        .replace("_filtered.csv", "")
    )
    LESS_results.at[Trial_ID, "Trial_ID"] = Trial_ID
    LESS_results_short.at[Trial_ID, "Trial_ID"] = Trial_ID

    (
        max_knee_flexion_deg_r,
        max_knee_flexion_deg_l,
        ic_knee_flexion_total,
        max_knee_flexion_total,
    ) = id_knee_flexion_error(Trial_ID, input_df_r, input_df_l, LESS_results, LESS_results_short)

    (
        max_hip_flexion_deg_r,
        max_hip_flexion_deg_l,
        ic_hip_flexion_total,
        max_hip_flexion_total,
    ) = id_hip_flexion_error(Trial_ID, input_df_r, input_df_l, LESS_results, LESS_results_short)

    ic_trunk_flexion_total, max_trunk_flexion_total = id_trunk_flexion_error(
        Trial_ID, input_df_r, LESS_results, LESS_results_short
    )

    ankle_PF_left, ankle_PF_right, ic_ankle_PF_total = id_ankle_plantarflexion_error(
        Trial_ID, coord_file, sampling_freq, coord_cutoff_freq, LESS_results, LESS_results_short
    )

    ic_timing_asymmetric_total = id_asymmetric_timing_error(
        Trial_ID, input_df_r, input_df_l, LESS_results, LESS_results_short
    )

    ic_landing_asymmetric_total = asymmetric_landing_error(
        Trial_ID, ankle_PF_right, ankle_PF_left, LESS_results, LESS_results_short
    )

    ic_trunk_lateral_total = id_lateral_trunk_flexion_error(
        Trial_ID, input_df_r, LESS_results, LESS_results_short
    )

    ic_knee_valgus_total, max_knee_valgus_total = id_knee_valgus_error(
        Trial_ID, input_df_r, input_df_l, LESS_results, LESS_results_short
    )

    ic_hip_wide_total, ic_hip_add_deg_r, ic_hip_add_deg_l = id_wide_stance_error(
        Trial_ID, input_df_r, input_df_l, LESS_results, LESS_results_short
    )

    ic_hip_narrow_total = id_narrow_stance_error(
        Trial_ID, ic_hip_add_deg_r, ic_hip_add_deg_l, LESS_results, LESS_results_short
    )

    ic_foot_inward_total = id_foot_inward_error(
        Trial_ID, input_df_r, input_df_l, LESS_results, LESS_results_short
    )

    ic_foot_outward_total = id_foot_outward_error(
        Trial_ID, input_df_r, input_df_l, LESS_results, LESS_results_short
    )

    max_asymmetric_loading_total = id_asymmetric_loading_error(
        Trial_ID, input_df_r, input_df_l, LESS_results, LESS_results_short
    )

    joint_displacement_total = id_joint_displacement_error(
        Trial_ID,
        max_hip_flexion_total,
        max_trunk_flexion_total,
        max_knee_flexion_total,
        max_knee_flexion_deg_r,
        max_knee_flexion_deg_l,
        max_hip_flexion_deg_r,
        max_hip_flexion_deg_l,
        LESS_results,
        LESS_results_short,
    )

    overall_impression_total = id_overall_impression_error(
        Trial_ID,
        max_hip_flexion_total,
        max_trunk_flexion_total,
        max_knee_flexion_total,
        max_knee_valgus_total,
        ic_knee_valgus_total,
        LESS_results,
        LESS_results_short,
    )

    LESS_total = (
        ic_knee_flexion_total
        + max_knee_flexion_total
        + ic_knee_valgus_total
        + max_knee_valgus_total
        + ic_trunk_lateral_total
        + ic_trunk_flexion_total
        + max_trunk_flexion_total
        + ic_hip_flexion_total
        + max_hip_flexion_total
        + ic_hip_wide_total
        + ic_hip_narrow_total
        + ic_foot_inward_total
        + ic_foot_outward_total
        + ic_ankle_PF_total
        + ic_timing_asymmetric_total
        + ic_landing_asymmetric_total
        + max_asymmetric_loading_total
        + joint_displacement_total
        + overall_impression_total
    )

    LESS_results.at[Trial_ID, "Total_Score"] = LESS_total
    LESS_results_short.at[Trial_ID, "Total_Score"] = LESS_total
    print(f"LESS_total value in row '{Trial_ID}': {LESS_total}")
    nan_count = LESS_results.loc[Trial_ID].isnull().sum()
    print(f"Number of NaNs in row '{Trial_ID}': {nan_count}")

    return LESS_results, LESS_results_short

def bilateral_process_file(
    file_name, kinematics_name, sampling_freq, cutoff_freq, coord_cutoff_freq, output_dir="."
):
    title = os.path.splitext(os.path.basename(file_name))[0]
    print(f"\nProcessing data from: {title}")

    (
        data,
        time,
        y_coords_left_toes,
        y_coords_left_heel,
        y_coords_right_toes,
        y_coords_right_heel,
    ) = load_coord_file(file_name)

    (
        left_toes_filtered,
        left_heel_filtered,
        right_toes_filtered,
        right_heel_filtered,
    ) = apply_butterworth_filter_coords(
        y_coords_left_toes,
        y_coords_left_heel,
        y_coords_right_toes,
        y_coords_right_heel,
        sampling_freq,
        coord_cutoff_freq,
    )

    # Left Side
    toes_first_minima_time, toes_first_minima_y_coord, lt_first_minima_index = (
        id_initialcontact_toes(left_toes_filtered, time)
    )
    heel_first_minima_time, lh_first_minima_index = id_initialcontact_heels(
        left_heel_filtered, time
    )
    y_min_index_l, y_min_index_l_time = compare_initialcontact_toe_vs_heel(
        toes_first_minima_time,
        lt_first_minima_index,
        heel_first_minima_time,
        lh_first_minima_index,
        time,
    )
    toe_off_index_l, toes_second_minima_time_l = id_takeoff_toes(
        left_toes_filtered, time, lt_first_minima_index
    )

    generate_plot_table_quality_check(
        time,
        left_heel_filtered,
        left_toes_filtered,
        y_min_index_l_time,
        toes_second_minima_time_l,
        title,
        "left",
        output_dir=output_dir,
    )

    left_kinematics, l_stance_coords = apply_stance_phase_to_motion_file(
        data,
        time,
        y_min_index_l,
        toe_off_index_l,
        kinematics_name,
        start_column=14,
        end_column=20,
    )

    l_filtered_data = apply_butterworth_filter(
        left_kinematics, sampling_freq, cutoff_freq
    )
    l_saved = save_filtered_data(l_filtered_data, title, side_of_body="left", output_dir=output_dir)

    # Right Side
    toes_first_minima_time_r, toes_first_minima_y_coord_r, rt_first_minima_index = (
        id_initialcontact_toes(right_toes_filtered, time)
    )
    heel_first_minima_time_r, rh_first_minima_index = id_initialcontact_heels(
        right_heel_filtered, time
    )
    y_min_index_r, y_min_index_r_time = compare_initialcontact_toe_vs_heel(
        toes_first_minima_time_r,
        rt_first_minima_index,
        heel_first_minima_time_r,
        rh_first_minima_index,
        time,
    )
    toe_off_index_r, toes_second_minima_time_r = id_takeoff_toes(
        right_toes_filtered, time, rt_first_minima_index
    )

    generate_plot_table_quality_check(
        time,
        right_heel_filtered,
        right_toes_filtered,
        y_min_index_r_time,
        toes_second_minima_time_r,
        title,
        "right",
        output_dir=output_dir,
    )

    right_kinematics, r_stance_coords = apply_stance_phase_to_motion_file(
        data,
        time,
        y_min_index_r,
        toe_off_index_r,
        kinematics_name,
        start_column=7,
        end_column=13,
    )

    r_filtered_data = apply_butterworth_filter(
        right_kinematics, sampling_freq, cutoff_freq
    )
    r_saved = save_filtered_data(r_filtered_data, title, side_of_body="right", output_dir=output_dir)

    return l_saved, r_saved


def main():
    base_dir = r"OpenCapData_71\OpenCapData_71b0b5e9-e46e-439b-b91a-22327e0ede92"
    kin_dir = os.path.join(base_dir, "OpenSimData", "Kinematics")
    marker_dir = os.path.join(base_dir, "MarkerData")
    output_dir = r"OpenCapData_71\OpenLESS_Results"
    os.makedirs(output_dir, exist_ok=True)

    trials = ["Stepoff-Leaning", "Stepoff-Upright"]
    sampling_freq = 60  # OpenCap camera data rate in these trials is 60 Hz
    cutoff_freq = 12
    coord_cutoff_freq = 12

    LESS_results = pd.DataFrame(columns=column_headers)
    LESS_results_short = pd.DataFrame(columns=column_headers_short)

    processed_files = {}

    # Step 3: Clip and filter
    for trial in trials:
        trc_file = os.path.join(marker_dir, f"{trial}.trc")
        mot_file = os.path.join(kin_dir, f"{trial}.mot")
        l_saved, r_saved = bilateral_process_file(
            trc_file,
            mot_file,
            sampling_freq,
            cutoff_freq,
            coord_cutoff_freq,
            output_dir=output_dir,
        )
        processed_files[trial] = {
            "left": l_saved,
            "right": r_saved,
            "trc": trc_file,
        }

    # Step 4: Automatically Score LESS
    print("\n" + "=" * 50)
    print("TOTAL LESS SCORES")
    print("=" * 50)
    for trial in trials:
        print(f"\nScoring trial: {trial}")
        LESS_results, LESS_results_short = process_bilateral_LESS(
            processed_files[trial]["right"],
            processed_files[trial]["left"],
            processed_files[trial]["trc"],
            LESS_results,
            LESS_results_short,
            sampling_freq,
            coord_cutoff_freq,
        )

    # Step 5: Export results
    full_csv = os.path.join(output_dir, "OpenCapData_71_less_scores_final.csv")
    short_csv = os.path.join(output_dir, "OpenCapData_71_less_short_final.csv")
    LESS_results.to_csv(full_csv, index=False)
    LESS_results_short.to_csv(short_csv, index=False)

    print("\n" + "=" * 50)
    print("LESS SHORT RESULTS:")
    print("=" * 50)
    pd.set_option("display.max_columns", 30)
    print(LESS_results_short.to_string())

    print("\nResults exported successfully:")
    print(f"Detailed scores: {full_csv}")
    print(f"Summary scores: {short_csv}")

if __name__ == "__main__":
    main()
