import matplotlib.pyplot as plt

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

# 扩展配色方案：每个调色板包含10种颜色（用于多模型区分）
PALETTES = {
    "Nature": {
        "primary": "#E64B35",
        "secondary": "#4DBBD5",
        "accent": "#00A087",
        "colors": ["#E64B35", "#4DBBD5", "#00A087", "#FFD700", "#8A2BE2",
                   "#FF69B4", "#20B2AA", "#CD5C5C", "#6A5ACD", "#FF8C00"]
    },
    "Lancet": {
        "primary": "#0072B5",
        "secondary": "#D55E00",
        "accent": "#F0E442",
        "colors": ["#0072B5", "#D55E00", "#F0E442", "#009E73", "#CC79A7",
                   "#56B4E9", "#E69F00", "#999999", "#66CCEE", "#BBBBBB"]
    },
    "Science": {
        "primary": "#56B4E9",
        "secondary": "#F0E442",
        "accent": "#009E73",
        "colors": ["#56B4E9", "#F0E442", "#009E73", "#D55E00", "#CC79A7",
                   "#0072B5", "#E69F00", "#999999", "#66CCEE", "#BBBBBB"]
    },
    "JAMA": {
        "primary": "#E69F00",
        "secondary": "#D55E00",
        "accent": "#0072B5",
        "colors": ["#E69F00", "#D55E00", "#0072B5", "#009E73", "#CC79A7",
                   "#56B4E9", "#F0E442", "#999999", "#66CCEE", "#BBBBBB"]
    },
    "Pastel": {
        "primary": "#FBB4AE",
        "secondary": "#B3CDE3",
        "accent": "#CCEBC5",
        "colors": ["#FBB4AE", "#B3CDE3", "#CCEBC5", "#DECBE4", "#FED9A6",
                   "#FFFFCC", "#E5D8BD", "#FDDAEC", "#F2F2F2", "#D9D9D9"]
    },
    "Vibrant": {
        "primary": "#FF6B6B",
        "secondary": "#4ECDC4",
        "accent": "#FFE66D",
        "colors": ["#FF6B6B", "#4ECDC4", "#FFE66D", "#1A535C", "#FF9F1C",
                   "#2EC4B6", "#E71D36", "#011627", "#FF9F1C", "#FDFFFC"]
    },
    "Dark": {
        "primary": "#2C3E50",
        "secondary": "#E74C3C",
        "accent": "#3498DB",
        "colors": ["#2C3E50", "#E74C3C", "#3498DB", "#2ECC71", "#F1C40F",
                   "#9B59B6", "#1ABC9C", "#E67E22", "#34495E", "#7F8C8D"]
    },
    "Earth": {
        "primary": "#8B5A2B",
        "secondary": "#6B8E23",
        "accent": "#CD853F",
        "colors": ["#8B5A2B", "#6B8E23", "#CD853F", "#556B2F", "#D2691E",
                   "#A0522D", "#8FBC8F", "#DEB887", "#F4A460", "#D2B48C"]
    }
}

def get_palette(name="Nature"):
    """
    返回配色字典，包含 primary, secondary, accent 以及 colors（10种颜色的列表）
    若名称不存在则返回 Nature
    """
    return PALETTES.get(name, PALETTES["Nature"])

def get_color_list(name="Nature"):
    """直接返回10种颜色的列表，便于绘图时循环使用"""
    return get_palette(name)["colors"]