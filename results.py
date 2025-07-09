import matplotlib.pyplot as plt
import numpy as np


def plot_connected(x, y, label, label_, **plot_kwargs):
    valid = ~np.isnan(y)
    idx = np.where(valid)[0]

    # Draw only the line segments (no label here)
    for i, j in zip(idx, idx[1:]):
        plt.plot(
            x[[i, j]],
            y[[i, j]],
            linestyle=plot_kwargs.get("linestyle", "-"),
            color=plot_kwargs.get("color", None),
            marker="",      # no marker on the segments
            # note: no label
        )

    # Draw the markers once, with the label
    plt.scatter(
        x[valid],
        y[valid],
        label=label,
        marker=plot_kwargs.get("marker", "o"),
        color=plot_kwargs.get("color", None),
    )
    plt.xlabel(label_)
#
#
# # your data
# intervals = np.arange(10, 101, 10)
# decen_alt_view = np.array([0.87968797, 0.76085761, 0.7016,      0.74684874, 0.74025974, np.nan, 0.67027027, np.nan, 0.66014,    np.nan])
# cen_list        = np.array([0.88748875, 0.82517867, 0.8232,     0.61449580, 0.72431730, 0.51472868, 0.36151079, 0.43121150, np.nan, 0.34693878])
# decen_list      = np.array([0.88092382, 0.83397471, 0.7928,     0.64810924, 0.61298701, 0.42635659, 0.33273381, 0.41889117, np.nan, 0.41      ])
# decent_alt_3    = np.array([0.66246625, 0.60066007, 0.5048,      np.nan,     0.53246753, np.nan,     np.nan,     0.48975410, np.nan, 0.49618321])
# decent_alt_9    = np.array([0.87218722, np.nan,     0.865,       np.nan,     0.80884265, np.nan,     0.81294964, np.nan,     np.nan, 0.70153061])
# random_         = np.array([0.31773177, 0.30198020, 0.3144,      0.31722689, 0.29908973, 0.31007752, 0.30395683, 0.30737705, 0.27816092, 0.33163265])
#
# plt.figure(figsize=(10,6))
#
# plot_connected(intervals, decen_list,      label='Decentralized',                   marker='^', linestyle='-', color='#2ca02c')
# plot_connected(intervals, cen_list,        label='Centralized',                     marker='s', linestyle='-', color='#d62728')
# plot_connected(intervals, decent_alt_9,    label='Decentralized (Local View = 9)', marker='o', linestyle='-', color='#9467bd')
# plot_connected(intervals, decent_alt_3,    label='Decentralized (Local View = 3)', marker='o', linestyle='-', color='#ff7f0e')
# plot_connected(intervals, decen_alt_view,  label='Decentralized (Local View = 5)', marker='o', linestyle='-', color='#1f77b4')
# plot_connected(intervals, random_,         label='Random',                          marker='o', linestyle='-', color='#8c564b')
#
# plt.ylim(0.2, 1.0)
# plt.xlabel('Intervals')
# plt.ylabel('Value')
# plt.grid(True)
# plt.legend(prop={'size': 8})
# plt.title('Ratio of Apples Picked by Approach')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

def interpolate_nans(x, y):
    """
    Linearly interpolates over NaNs in the y array.
    """
    x = np.array(x)
    y = np.array(y, dtype=np.float64)
    mask = np.isnan(y)
    if mask.all():
        return y  # All values are NaN, nothing to interpolate
    y[mask] = np.interp(x[mask], x[~mask], y[~mask])
    return y

def plot_apple_picking(widths, series_dict, title, label_x, label_y):
    """
    Plots apple-picking ratios for different approaches.
    NaNs in data are interpolated to connect lines.
    """
    plt.figure()
    for label, ratios in series_dict.items():
        y_interp = interpolate_nans(widths, ratios)
        plt.plot(widths, y_interp, marker='o', label=label)

    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    plt.legend()
    plt.xticks(widths)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# # Example usage for l=10, n=2
# widths = [1, 2, 3, 4, 5]
# series_l10 = {
#     'Decentralized': [0.8763876387638764, 0.7455745574557455, 0.6747674767476748, 0.5955595559555955, 0.5142514251425142],
#     'Centralized': [0.864986498649865, 0.5931593159315932, 0.3812237552489502, 0.2516496700659868, 0.2076207620762076],
#     'Decentralized (local view=9)': [0.8718871887188719, 0.7428742874287428, 0.6858685868586859, 0.549054905490549, 0.5334533453345335],
#     'Random': [0.3165316531653165, 0.15356928614277143, 0.10561056105610561, 0.06808638272345531, 0.056105610561056105],
# }
# plot_apple_picking(widths, series_l10, 'Apple-Picking Ratio vs Width (l=10, n=2)')
#
# # Example usage for l=20, n=4
# series_l20 = {
#     'Decentralized': [0.7948294829482948, 0.6771177117711771, 0.49, 0.42629262926292627, 0.3564356435643564],
#     'Centralized': [0.8564356435643564, 0.49, 0.35533553355335534, 0.27447744774477445, 0.21],
#     'Random': [0.2887788778877888, 0.14796479647964797, 0.10891089108910891, 0.0725673446948873, 0.056105610561056105],
# }
# plot_apple_picking(widths, series_l20, 'Apple-Picking Ratio vs Width (l=20, n=4)')
#

# series_l20_new = {
#     'Decentralized (New)':  [0.8338790931989924, 0.723376049491825, None, None, None],
#     'Centralized (New)': [0.8230176485210042, 0.632222849614154, 0.6079388050444491, 0.5211638552705838, 0.44191275670813973],
#     'Random (New)': [0.5556246888999502, 0.2979082110521386, 0.21796587487453997, 0.15968414125904803, 0.13237283888749687],
#     'Decentralized (Old)':  [0.7948294829482948, None, None, None, None],
#     'Centralized (Old)': [0.8564356435643564, 0.5145684442001099, 0.3256325632563256, 0.2726772952171523,  0.264026402640264],
#     'Random (Old)': [0.3030803080308031, 0.16666666666666666, 0.12046204620462046, 0.09290819131390875, 0.0748074807480748]
# }
# plot_apple_picking(widths, series_l20_new, 'Apple Ratio vs Width (l=20, n=0.2*l*w) (New vs Old Orchard)')


# lengths = [10, 20, 40, 50, 60, 70, 80, 90, 100]
# random_1d_test = {
#     'Random': [0.2328, 0.1963549088727218, 0.18253333333333333, 0.176375, 0.1866753331166721, 0.17571341833636916, 0.17420733879586747, 0.1811293082375947, 0.1741765164696706]
# }
# plot_apple_picking(lengths, random_1d_test, 'Apple Ratio vs Orchard Length (1D)', "Length")
#
#
# widths = [1, 2, 3, 4, 5, 10]
# random_2d_test = {
#     'Random': [0.1963549088727218, 0.174375, 0.1814375, 0.17782923643272772, 0.1814472578903156, 0.1804381576677592]
# }
# plot_apple_picking(widths, random_2d_test, 'Apple Ratio vs Orchard Width (Length = 20) (2D)', "Width")

# cen_new = [0.7356621480709072, 0.7249936045024303, np.nan, np.nan, 0.7141682934347067]
# cen_new_apples_per_second = [0.0726, 0.071, np.nan, np.nan, 0.07]
# random_new_per_second = [0.0322, 0.032, np.nan, np.nan, 0.030959999999999297]
# decen_new = [0.6454280155642024, 0.5900052328623757, np.nan, np.nan, 0.54]
# decen_new_apples_per_second = [0.06635, 0.056375, np.nan, np.nan, 0.055]
# decen_alt_vision = [0.6014354066985645, 0.5811504879301489, np.nan, np.nan, 0.6158281964605289]
# decen_alt_vision_apples_per_second = [0.06285, 0.056575, np.nan, np.nan, 0.06194000000000391]
# random = [0.3326612903225806, 0.3211642821904292, np.nan, np.nan, 0.31526702402223544]
#
#
# apples_picked_random = [330, 651, np.nan, np.nan, 1588]
# apples_picked_cen = [716, 1464, np.nan, np.nan, 3538]
# apples_picked_decen = [608, 1190, np.nan, np.nan, 2719]
# apples_picked_decen_alt_vision = [669, 1208, np.nan, np.nan, 3075]
#
# apples_picked_per_agent_random = [165.0, 162.75, np.nan, np.nan, 158.8]
# apples_picked_per_agent_cen = [358.0, 366.0, np.nan, np.nan, 353.8]
# apples_picked_per_agent_decen = [304.0, 297.5, np.nan, np.nan, 271.9]
# apples_picked_per_agent_decen_alt_vision = [334.5, 302.0, np.nan, np.nan, 307.5]
#
#
# lengths = [10, 20, 30, 40, 50]
# new_series = {
#     'Centralized': cen_new,
#     'Decentralized': decen_new,
#     'Random': random,
#     'Decentralized (alt vision)': decen_alt_vision,
#
# }
#
# new_series_per_second = {
#     'Centralized (apples/sec)': cen_new_apples_per_second,
#     'Decentralized (apples/sec)': decen_new_apples_per_second,
#     'Decentralized (alt vision, apples/sec)': decen_alt_vision_apples_per_second,
#     'Random (apples/sec)': random_new_per_second,
# }
#
# new_series_apples = {
#     'Centralized': apples_picked_cen,
#     'Decentralized': apples_picked_decen,
#     'Random': apples_picked_random,
#     'Decentralized (alt vision)': apples_picked_decen_alt_vision,
#
# }
#
# new_series_apples_per_agent = {
#     'Centralized (apples/agent)': apples_picked_per_agent_cen,
#     'Decentralized (apples/agent)': apples_picked_per_agent_decen,
#     'Decentralized (alt vision, apples/agent)': apples_picked_per_agent_decen_alt_vision,
#     'Random (apples/agent)': apples_picked_per_agent_random,
# }
#
# old_series_apples = {
#     'Centralized': [2982, 1535, 1015, 572, 572],
#     'Decentralized': [2921, 1516, 991, 617, 472],
#     'Random': [1080, 604, 353, 286, 226],
# }
#
# old_series_apples_per_agent = {
#     'Centralized (apples/agent)': [1491.0, 383.75, 169.16666666666666, 71.5,
#                                    57.2],
#     'Decentralized (apples/agent)': [1460.5, 379.0, 165.16, 77.125, 47.2],
#     'Random (apples/agent)': [540.0, 151.0, 58.833333333333336, 35.75, 22.6],
# }

total = {
    'Centralized': [40053, 7942, 3942, 1954, 779, 395],
    'Decentralized': [40053, 7942, 3942, 1954, 779, 395],
    'Random': [39984, 8138, 4118, 2061, 796, 387],
    'Nearest': [39984, 8138, 4118, 2061, 796, 387]
}

picked = {
    'Centralized': [23822, 5788, 2883, 1375, 586, 293],
    'Decentralized': [23252, 5069, 2255, 1015, 371, 197],
    'Random': [13291, 2616, 1336, 677, 248, 118],
    'Nearest': [23719, 5455, 2744, 1382, 513, 230]
}


picked_per_agent = {
    'Centralized': [5955.666666666667, 1447.0, 720.9166666666666, 343.75, 146.58333333333334, 73.25],
    'Decentralized': [5813.083333333333, 1267.25, 563.9166666666666, 253.83333333333334, 92.75, 49.25],
    'Random': [3322.75, 654.0, 334.0, 169.25, 62.0, 29.5],
    'Nearest': [5929.75, 1363.75, 686.0, 345.5, 128.25, 57.5]
}


ratio_per_agent = {
    'Centralized': [0.5947690805259811, 0.7287496030168849, 0.7314163675430706, 0.7037494486941022, 0.7521125277030789, 0.74081557376207],
    'Decentralized': [0.5805287925152777, 0.6382155616141194, 0.5721329740276379, 0.5197566014676797, 0.4759781255844248, 0.49782768292198876],
    'Random': [0.3324079631852741, 0.3214549029245515, 0.3244293346284604, 0.3284813197476953, 0.31155778894472363, 0.3049095607235142],
    'Nearest': [0.5932122849139656, 0.670312115999017, 0.6663428848955804, 0.6705482775351771, 0.6444723618090452, 0.5943152454780362]
}


mean_distance = {
    'Centralized': [4.5942875, 3.5876375, 3.7823875, 3.65905, 4.04380625, 4.377375],
    'Decentralized': [3.7122, 3.5325875, 3.41985, 3.1226125, 3.5565625, 3.57595],
    'Random': [3.00668125, 3.00668125, 3.00668125, 3.00668125, 3.00668125, 3.00668125]
}


time = [1, 5, 10, 20, 50, 100]

#
# plot_apple_picking(time, total, 'Total Apples Spawned', "Seconds for 1 Apple to Spawn Per Agent", "Apples")
# plot_apple_picking(time, picked, 'Total Apples Picked', "Seconds for 1 Apple to Spawn Per Agent", "Apples")
# plot_apple_picking(time, picked_per_agent, 'Apples Picked per Agent', "Seconds for 1 Apple to Spawn Per Agent", "Apples")
# plot_apple_picking(time, ratio_per_agent, 'Ratio of Apples Picked per Agent', "Seconds for 1 Apple to Spawn Per Agent", "Ratio of Apples Picked Per Agent")
#

# Centralized (256)

[3942, 3942, 3942, 3942]
[1554, 2550, 3455, 3846]
[388.5, 637.6666666666666, 863.8333333333334, 961.5833333333334]
[0.3941289182694585, 0.6469776265610848, 0.8764985080491904, 0.9756506508111151]
[3.85418125, 3.0570625, 3.4809375, 4.97315625]

# DC local view (256)
[3942, 4082, 4082, 4082]
[1184, 2169, 3120, 3421]
[296.0, 542.4166666666666, 780.1666666666666, 855.25]
[0.3003259685608622, 0.5314451671547817, 0.7644330340994937, 0.8379132259446797]
[2.39394375, 2.360975, 2.85360625, 2.80659375]
[0.2756, 0.4895, 1.1073, 5.5709]

total_apples = {
    'Random': [4118, 4118, 4118, 4118],
    'Nearest': [4118, 4118, 4118, 4118],
    'Centralized': [4082, 4082, 4082, 4082],
    'Decentralized': [4082, 4082, 4082, 4082],
    'Decentralized (local view = 9)': [4082, 4086, 4082, 4082]
}

picked = {
    'Random': [1128, 2038, 3026, 3602],
    'Nearest': [1616, 2443, 3462, 3967],
    'Centralized': [1861, 2678, 3535, 3950],
    'Decentralized': [1338, 2214, 3228, 3765],
    'Decentralized (local view = 9)': [1590, 2438, 3170, 3345]
}

picked_per_agent = {
    'Random': [282.0, 509.5, 756.5, 900.5],
    'Nearest': [404.0, 610.75, 865.5, 991.75],
    'Centralized': [465.25, 669.5833333333334, 883.9166666666666,
                    987.6666666666666],
    'Decentralized': [334.6666666666667, 553.5, 807.0833333333334,
                      941.4166666666666],
    'Decentralized (local view = 9)': [397.6666666666667, 609.5,
                                       792.6666666666666, 836.3333333333334]
}

ratio_per_agent = {
    'Random': [0.27391937833899954, 0.49490043710539094, 0.7348227294803302,
               0.8746964545896067],
    'Nearest': [0.39242350655658087, 0.593249150072851, 0.8406993686255464,
                0.9633317144244778],
    'Centralized': [0.4558220191245163, 0.6560976539963072, 0.86609967282057,
                    0.9677207510707194],
    'Decentralized': [0.3278783281917137, 0.542208845351332, 0.7908430202638885,
                      0.9222574267372012],
    'Decentralized (local view = 9)': [0.38960184627446987, 0.5966261473579757,
                                       0.7766747740612834, 0.8197694380897346]
}

mean_distance = {
    'Random': [3.00668125, 3.00668125, 3.00668125, 3.00668125],
    'Nearest': [1.468375, 1.8220375, 2.20399375, 2.263225],
    'Centralized': [3.325525, 4.02491875, 3.80764375, 4.096375],
    'Decentralized': [3.6287625, 3.1438875, 3.769647916666667,
                      4.194768750000001],
    'Decentralized (local view = 9)': [2.39394375, 2.7840375, 2.793441666666667,
                                       2.7789]
}

apples_per_second = {
    'Nearest': [0.2502, 0.4262, 0.6594, 0.7967],
    'Centralized': [0.2235, 0.362, 0.5621, 0.7167],
    'Decentralized': [0.2743333333333333, 0.4801333333333333,
                      0.8906999999999999, 2.1884],
    'Decentralized (local view = 9)': [0.2756, 0.419, 1.0231, 6.9823],
    'Random': [0.3414, 0.7611, 2.3368, 6.7474]
}

timesteps = [1, 2.5, 10, 50]
plot_apple_picking(timesteps, total_apples, 'Total Apples Available',
                   'Timesteps', 'Number of Apples')
plot_apple_picking(timesteps, picked, 'Total Apples Picked', 'Timesteps',
                   'Number of Apples')
plot_apple_picking(timesteps, picked_per_agent, 'Apples Picked per Agent',
                   'Timesteps', 'Number of Apples')
plot_apple_picking(timesteps, ratio_per_agent, 'Ratio of Apples Picked Per Agent',
                   'Timesteps', 'Ratio')
plot_apple_picking(timesteps, mean_distance, 'Mean Distance Between Agents',
                   'Timesteps', 'Distance')
plot_apple_picking(timesteps, apples_per_second,
                   'Average Number of Apples Available Per Agent Per Second', 'Timesteps', 'Apples/Second')

total_apples = {
    'Centralized': [3334, 3334, 3334, 3334],
    'Decentralized': [3334, 3334, 3334, 3334]
}

picked = {
    'Centralized': [3039, 2211, 1551, 843],
    'Decentralized': [3153, 2523, 2078, 1247]
}

picked_per_agent = {
    'Centralized': [1519.5, 1105.6666666666667, 775.6666666666666, 421],
    'Decentralized': [1576.8333333333333, 1261.6666666666667, 1039.0, 623]
}

ratio_per_agent = {
    'Centralized': [0.9115176964607078, 0.6632673465306939, 0.4653069386122775, 0.2529494101179764],
    'Decentralized': [0.9459108178364328, 0.7568486302739452, 0.37]
}

mean_distance = {
    'Centralized': [5.979433333333333, 2.2759470930321894, 1.5527212381952598, 2.934707244602919],
    'Decentralized': [4.994266666666667, 3.5691879712044767, 3.0286420714347635, 4.2]
}

apples_per_second = {
    'Centralized': [0.5959333333333333, 0.7415333333333334, 0.8295666666666667, 0.9055333333333334],
    'Decentralized': [0.5569333333333334, 0.711, 0.7890666666666667, 0.88]
}

width = [1, 3, 5, 10]
plot_apple_picking(width, total_apples, 'Total Apples Available',
                   'Timesteps', 'Number of Apples')
plot_apple_picking(width, picked, 'Total Apples Picked', 'Timesteps',
                   'Number of Apples')
plot_apple_picking(width, picked_per_agent, 'Apples Picked per Agent',
                   'Timesteps', 'Number of Apples')
plot_apple_picking(width, ratio_per_agent,
                   'Ratio of Apples Picked Per Agent',
                   'Timesteps', 'Ratio')
plot_apple_picking(width, mean_distance, 'Mean Distance Between Agents',
                   'Timesteps', 'Distance')
plot_apple_picking(width, apples_per_second,
                   'Average Number of Apples Available Per Agent Per Second',
                   'Timesteps', 'Apples/Second')


# NN experiments

# total_apples = {
#     'Centralized': [3334, 3334, 3334, 3334],
#     'Decentralized': [3334, 3334, 3334, 3334]
# }
#
# picked = {
#     'Centralized': [3207, 3207, 3207, 3176],
#     'Decentralized': [3158, 3207, 3207, 3207]
# }

# picked_per_agent = {
#     'Centralized': [662, 2044, 2643, 2609],
#     'Decentralized': [1384, 2458, 2493, 2613]
# }


# HIDDEN DIMENSIONS
ratio_per_agent = {
    'Centralized': [0.20660973317241194, 0.6374676595161487, 0.8244855353157848, 0.8214],
    'Decentralized': [0.438, 0.7664176776030556, 0.7774085233627424, 0.8147826641866422]
}

mean_distance = {
    'Centralized': [0.18866666666666668, 2.730641666666667, 4.068766666666666, 4.0489],
    'Decentralized': [2.396, 2.9726166666666667, 2.9828499999999996, 3.3261416666666666]
}

hidden_dimensions = [4, 16, 32, 64]


plot_apple_picking(hidden_dimensions, ratio_per_agent,
                   'Ratio of Apples Picked Per Agent',
                   'Timesteps', 'Ratio')
plot_apple_picking(hidden_dimensions, mean_distance, 'Mean Distance Between Agents',
                   'Timesteps', 'Distance')


# LAYERS
ratio_per_agent = {
    'Centralized': [0.21784811964723358, 0.503463476070529, 0.5768778731320733, 0.6325187859555231],
    'Decentralized': [0.45065753339577347, 0.6182384422321842, 0.6509583761172931, 0.676007556675063]
}

mean_distance = {
    'Centralized': [0.7660499999999999, 2.381535370602413, 2.8785833333333333, 3.0528166666666667],
    'Decentralized': [3.105775, 3.5018166666666666, 4.706575, 4.375]
}

layers = [1, 2, 3, 4]

plot_apple_picking(hidden_dimensions, ratio_per_agent,
                   'Ratio of Apples Picked Per Agent',
                   'Timesteps', 'Ratio')
plot_apple_picking(hidden_dimensions, mean_distance, 'Mean Distance Between Agents',
                   'Timesteps', 'Distance')

# C
# [3207, 3176, 3207, 3207]
# [696, 1599, 1850, 2028]
# [174.16666666666666, 399.75, 462.5, 507.1666666666667]


# DC
# [3248, 3207, 3207, 3176]
# [1464, 1983, 2087, 2147]
# [366.0, 495.8333333333333, 521.9166666666666, 536.75]



# WIDTH

# total_apples = {
#     'Centralized': [3196, 3199, 3162, 3244],
#     'Decentralized': [3196, 3199, 3162, 3244]
# }
#
# picked = {
#     'Centralized': [2684, 2145, 1166, 591],
#     'Decentralized': [2670, 2180, 1798, 1099]
# }

ratio_per_agent = {
    'Centralized': [0.8398596256558792, 0.6706103154347121, 0.36863358634143784, 0.1822164274427124],
    'Decentralized': [0.8354606297728423, 0.6815443674495469, 0.5684422028905476, 0.33883804216722524]
}

mean_distance = {
    'Centralized': [6.248266666666666, 3.7923009200006845, 0.8779087957639603, 0.3892876387920709],
    'Decentralized': [6.0995, 3.3080265823612827, 2.7073820399272877, 1.8412929058627483]
}

widths = [1, 3, 5, 8]

plot_apple_picking(widths, ratio_per_agent,
                   'Ratio of Apples Picked Per Agent',
                   'Timesteps', 'Ratio')
plot_apple_picking(widths, mean_distance, 'Mean Distance Between Agents',
                   'Timesteps', 'Distance')

# LENGTH

# Fix agents, DC
#
# [3186, 3213, 3217]
# [2316, 1817, 1165]
# [1158.1666666666667, 908.5, 582.5]
#
#
# [0.6753666666666667, 0.9095, 1.1376]


# Fix agents, C

# [3186, 3213, 3217]
# [2336, 1944, 1366]
# [1168.3333333333333, 972.0, 683.1666666666666]
#
#
# [0.6756666666666667, 0.8431666666666667, 1.0590333333333335]


ratio_per_agent = {
    'Centralized': [0.8398596256558792, 0.7334991477691171, 0.6050709717838781, 0.4246823196727571],
    'Decentralized': [0.8354606297728423, 0.7269721134525585, 0.5654281251171368, 0.3620783459009132]
}

mean_distance = {
    'Centralized': [6.248266666666666, 6.2, 7.718800000000001, 10.460733333333335],
    'Decentralized': [6.0995, 5.945666666666667, 8.027533333333333, 9.383933333333333]
}

lengths = [10, 15, 20, 30]

plot_apple_picking(lengths, ratio_per_agent,
                   'Ratio of Apples Picked Per Agent',
                   'Timesteps', 'Ratio')
plot_apple_picking(widths, mean_distance, 'Mean Distance Between Agents',
                   'Timesteps', 'Distance')




