"""
标准PDF卡片数据库
包含常见矿物的标准衍射数据（2theta和相对强度）
基于Cu Kα辐射 (λ=1.5406 Å)
"""
from dataclasses import dataclass, field
from typing import List, Tuple
import json
import os


@dataclass
class PDFCard:
    """PDF标准卡片"""
    name: str
    formula: str
    crystal_system: str
    peaks: List[Tuple[float, float]]  # (2theta, intensity)
    rir: float = 1.0  # Reference Intensity Ratio
    card_number: str = ""
    
    @property
    def num_peaks(self) -> int:
        return len(self.peaks)
    
    @property
    def two_theta_list(self) -> List[float]:
        return [p[0] for p in self.peaks]
    
    @property
    def intensity_list(self) -> List[float]:
        return [p[1] for p in self.peaks]


def _build_pdf_database() -> List[PDFCard]:
    """构建标准PDF卡片数据库"""
    cards = []
    
    cards.append(PDFCard(
        name="石英 (Quartz)",
        formula="SiO2",
        crystal_system="三方晶系",
        card_number="01-075-0443",
        rir=3.07,
        peaks=[
            (20.87, 4.5), (26.64, 100.0), (36.54, 12.5), (39.46, 7.0),
            (40.29, 3.5), (42.45, 3.1), (45.79, 1.7), (49.93, 6.5),
            (50.14, 8.0), (54.85, 1.7), (55.32, 1.6), (59.97, 2.2),
            (60.38, 2.8), (64.07, 1.1), (67.74, 2.5), (68.12, 2.7),
        ]
    ))
    
    cards.append(PDFCard(
        name="正长石 (Orthoclase)",
        formula="KAlSi3O8",
        crystal_system="单斜晶系",
        card_number="01-075-0133",
        rir=2.1,
        peaks=[
            (20.94, 30.0), (23.58, 100.0), (27.54, 20.0), (29.98, 15.0),
            (30.48, 25.0), (34.82, 10.0), (35.78, 8.0), (41.32, 7.0),
            (42.86, 5.0), (45.24, 6.0), (51.24, 4.0), (54.98, 3.0),
            (60.08, 4.0), (64.12, 2.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="钠长石 (Albite)",
        formula="NaAlSi3O8",
        crystal_system="三斜晶系",
        card_number="01-074-1014",
        rir=2.3,
        peaks=[
            (13.96, 20.0), (21.98, 30.0), (23.78, 100.0), (24.44, 25.0),
            (27.78, 15.0), (29.68, 20.0), (30.18, 10.0), (32.08, 12.0),
            (34.98, 8.0), (42.14, 5.0), (46.54, 4.0), (51.88, 3.0),
            (57.64, 2.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="钙长石 (Anorthite)",
        formula="CaAl2Si2O8",
        crystal_system="三斜晶系",
        card_number="01-071-0751",
        rir=2.0,
        peaks=[
            (12.24, 15.0), (21.94, 20.0), (23.74, 25.0), (24.58, 100.0),
            (27.78, 12.0), (28.28, 18.0), (29.88, 10.0), (30.68, 15.0),
            (33.48, 8.0), (35.88, 6.0), (41.88, 5.0), (46.84, 4.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="方解石 (Calcite)",
        formula="CaCO3",
        crystal_system="三方晶系",
        card_number="01-072-1937",
        rir=3.0,
        peaks=[
            (23.04, 8.0), (29.40, 100.0), (35.96, 12.0), (39.42, 14.0),
            (43.16, 10.0), (47.48, 18.0), (48.50, 22.0), (50.78, 3.0),
            (56.46, 2.0), (57.42, 6.0), (60.58, 4.0), (61.42, 1.0),
            (64.64, 3.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="白云石 (Dolomite)",
        formula="CaMg(CO3)2",
        crystal_system="三方晶系",
        card_number="01-070-1934",
        rir=2.5,
        peaks=[
            (20.96, 10.0), (24.18, 15.0), (30.96, 100.0), (34.28, 8.0),
            (35.58, 12.0), (37.36, 6.0), (41.14, 8.0), (44.96, 10.0),
            (46.38, 14.0), (50.66, 4.0), (51.28, 6.0), (55.08, 2.0),
            (60.04, 3.0), (61.68, 2.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="黄铁矿 (Pyrite)",
        formula="FeS2",
        crystal_system="等轴晶系",
        card_number="01-071-2219",
        rir=6.1,
        peaks=[
            (28.52, 100.0), (33.10, 30.0), (37.08, 15.0), (40.78, 5.0),
            (43.68, 10.0), (47.24, 8.0), (50.46, 20.0), (53.44, 6.0),
            (56.28, 4.0), (59.00, 8.0), (61.62, 3.0), (64.14, 4.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="赤铁矿 (Hematite)",
        formula="Fe2O3",
        crystal_system="三方晶系",
        card_number="01-073-0603",
        rir=5.0,
        peaks=[
            (24.14, 30.0), (33.16, 100.0), (35.62, 25.0), (40.86, 10.0),
            (49.48, 15.0), (54.10, 5.0), (57.56, 10.0), (62.44, 8.0),
            (64.02, 6.0), (69.44, 4.0), (71.96, 3.0), (75.14, 2.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="磁铁矿 (Magnetite)",
        formula="Fe3O4",
        crystal_system="等轴晶系",
        card_number="01-074-1910",
        rir=4.6,
        peaks=[
            (30.08, 25.0), (35.46, 100.0), (43.06, 35.0), (53.42, 12.0),
            (56.96, 25.0), (62.58, 10.0), (67.60, 3.0), (71.76, 3.0),
            (74.66, 2.0), (78.88, 2.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="高岭石 (Kaolinite)",
        formula="Al2Si2O5(OH)4",
        crystal_system="三斜晶系",
        card_number="01-078-1996",
        rir=1.8,
        peaks=[
            (12.36, 100.0), (20.04, 40.0), (21.26, 15.0), (24.90, 30.0),
            (26.60, 10.0), (35.08, 15.0), (36.50, 10.0), (38.48, 8.0),
            (40.94, 5.0), (45.46, 5.0), (50.34, 3.0), (54.98, 2.0),
            (60.04, 2.0), (62.64, 1.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="蒙脱石 (Montmorillonite)",
        formula="(Na,Ca)0.33(Al,Mg)2Si4O10(OH)2·nH2O",
        crystal_system="单斜晶系",
        card_number="01-076-0933",
        rir=1.5,
        peaks=[
            (6.50, 100.0), (19.80, 30.0), (24.90, 10.0), (27.60, 8.0),
            (34.90, 15.0), (37.60, 5.0), (45.10, 4.0), (53.80, 3.0),
            (60.10, 2.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="伊利石 (Illite)",
        formula="K0.65Al2.0Al0.65Si3.35O10(OH)2",
        crystal_system="单斜晶系",
        card_number="01-078-1518",
        rir=1.6,
        peaks=[
            (8.84, 100.0), (17.74, 20.0), (19.84, 25.0), (26.64, 30.0),
            (29.84, 8.0), (35.04, 10.0), (36.56, 8.0), (45.24, 5.0),
            (50.04, 3.0), (55.08, 2.0), (60.14, 2.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="绿泥石 (Chlorite)",
        formula="(Mg,Fe)5Al2Si3O10(OH)8",
        crystal_system="单斜晶系",
        card_number="01-071-1232",
        rir=1.4,
        peaks=[
            (6.22, 100.0), (12.48, 20.0), (18.76, 15.0), (24.96, 25.0),
            (28.48, 10.0), (30.98, 12.0), (35.06, 15.0), (36.58, 8.0),
            (39.58, 6.0), (42.56, 5.0), (45.86, 4.0), (50.16, 3.0),
            (54.98, 2.0), (60.10, 2.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="石膏 (Gypsum)",
        formula="CaSO4·2H2O",
        crystal_system="单斜晶系",
        card_number="01-070-0982",
        rir=2.8,
        peaks=[
            (11.62, 100.0), (20.72, 15.0), (23.42, 30.0), (29.12, 25.0),
            (31.12, 8.0), (33.36, 10.0), (35.86, 6.0), (40.72, 5.0),
            (43.46, 8.0), (47.66, 6.0), (49.16, 4.0), (55.16, 3.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="硬石膏 (Anhydrite)",
        formula="CaSO4",
        crystal_system="正交晶系",
        card_number="01-072-0916",
        rir=3.2,
        peaks=[
            (20.80, 8.0), (23.38, 12.0), (25.46, 100.0), (27.86, 15.0),
            (31.04, 10.0), (33.56, 8.0), (35.64, 6.0), (39.54, 5.0),
            (42.36, 8.0), (47.86, 4.0), (48.96, 3.0), (54.76, 3.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="萤石 (Fluorite)",
        formula="CaF2",
        crystal_system="等轴晶系",
        card_number="01-071-1928",
        rir=2.6,
        peaks=[
            (28.26, 100.0), (40.72, 20.0), (50.36, 10.0), (58.56, 8.0),
            (65.84, 3.0), (72.44, 2.0), (78.52, 1.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="方铅矿 (Galena)",
        formula="PbS",
        crystal_system="等轴晶系",
        card_number="01-072-1413",
        rir=8.5,
        peaks=[
            (26.02, 100.0), (30.10, 20.0), (43.14, 30.0), (51.02, 8.0),
            (53.58, 10.0), (62.74, 4.0), (70.92, 3.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="闪锌矿 (Sphalerite)",
        formula="ZnS",
        crystal_system="等轴晶系",
        card_number="01-072-0010",
        rir=5.5,
        peaks=[
            (28.52, 100.0), (33.10, 20.0), (47.52, 25.0), (56.32, 12.0),
            (59.00, 5.0), (69.48, 8.0), (76.78, 4.0), (81.84, 3.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="黄铜矿 (Chalcopyrite)",
        formula="CuFeS2",
        crystal_system="四方晶系",
        card_number="01-071-2030",
        rir=4.5,
        peaks=[
            (29.34, 100.0), (30.10, 8.0), (36.50, 12.0), (44.84, 5.0),
            (47.88, 8.0), (49.32, 10.0), (54.20, 3.0), (56.64, 6.0),
            (59.02, 2.0), (63.48, 3.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="辉钼矿 (Molybdenite)",
        formula="MoS2",
        crystal_system="六方晶系",
        card_number="01-073-1508",
        rir=7.5,
        peaks=[
            (14.38, 100.0), (29.04, 15.0), (32.68, 20.0), (39.54, 8.0),
            (43.88, 5.0), (47.62, 6.0), (49.78, 3.0), (56.18, 4.0),
            (58.22, 3.0), (60.18, 2.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="金红石 (Rutile)",
        formula="TiO2",
        crystal_system="四方晶系",
        card_number="01-073-1232",
        rir=4.8,
        peaks=[
            (27.44, 100.0), (36.08, 30.0), (39.20, 20.0), (41.24, 5.0),
            (44.06, 8.0), (54.32, 15.0), (56.60, 8.0), (62.76, 4.0),
            (64.08, 6.0), (68.98, 3.0), (69.78, 3.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="锐钛矿 (Anatase)",
        formula="TiO2",
        crystal_system="四方晶系",
        card_number="01-071-1166",
        rir=4.3,
        peaks=[
            (25.28, 100.0), (37.80, 12.0), (38.58, 15.0), (48.04, 12.0),
            (53.88, 3.0), (55.06, 4.0), (62.68, 5.0), (68.76, 2.0),
            (70.32, 3.0), (75.04, 2.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="刚玉 (Corundum)",
        formula="Al2O3",
        crystal_system="三方晶系",
        card_number="01-071-1683",
        rir=3.5,
        peaks=[
            (25.56, 20.0), (35.16, 100.0), (37.78, 25.0), (43.36, 8.0),
            (52.56, 10.0), (57.50, 15.0), (61.30, 8.0), (66.52, 5.0),
            (68.38, 3.0), (74.62, 2.0), (76.90, 3.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="橄榄石 (Olivine)",
        formula="(Mg,Fe)2SiO4",
        crystal_system="正交晶系",
        card_number="01-071-1248",
        rir=2.4,
        peaks=[
            (17.84, 20.0), (22.84, 30.0), (25.24, 15.0), (30.08, 100.0),
            (32.86, 10.0), (36.14, 8.0), (38.74, 12.0), (41.24, 6.0),
            (45.26, 8.0), (49.16, 4.0), (50.76, 6.0), (52.86, 3.0),
            (55.16, 4.0), (60.36, 3.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="石榴子石 (Garnet)",
        formula="Mg3Al2Si3O12",
        crystal_system="等轴晶系",
        card_number="01-073-1212",
        rir=2.8,
        peaks=[
            (20.04, 15.0), (23.46, 20.0), (27.58, 100.0), (29.76, 8.0),
            (33.86, 12.0), (37.32, 25.0), (41.04, 8.0), (42.66, 10.0),
            (45.58, 6.0), (48.98, 15.0), (52.86, 5.0), (57.12, 3.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="电气石 (Tourmaline)",
        formula="NaMg3Al6(BO3)3Si6O18(OH)4",
        crystal_system="三方晶系",
        card_number="01-072-0136",
        rir=1.9,
        peaks=[
            (13.86, 25.0), (20.94, 30.0), (22.46, 100.0), (24.78, 15.0),
            (28.36, 12.0), (30.16, 8.0), (34.66, 10.0), (37.76, 6.0),
            (40.86, 5.0), (45.06, 4.0), (48.86, 3.0), (53.66, 2.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="滑石 (Talc)",
        formula="Mg3Si4O10(OH)2",
        crystal_system="单斜晶系",
        card_number="01-071-1180",
        rir=1.3,
        peaks=[
            (9.66, 100.0), (19.22, 20.0), (24.26, 15.0), (28.58, 12.0),
            (32.08, 8.0), (35.08, 10.0), (36.78, 6.0), (39.98, 4.0),
            (45.68, 3.0), (50.18, 2.0), (54.98, 2.0), (60.18, 1.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="蛇纹石 (Serpentine)",
        formula="Mg3Si2O5(OH)4",
        crystal_system="单斜晶系",
        card_number="01-072-1217",
        rir=1.7,
        peaks=[
            (12.18, 100.0), (18.36, 20.0), (24.52, 25.0), (27.44, 15.0),
            (30.18, 10.0), (35.26, 12.0), (36.98, 8.0), (40.16, 5.0),
            (45.76, 4.0), (50.46, 3.0), (55.26, 2.0), (60.26, 2.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="海泡石 (Sepiolite)",
        formula="Mg4Si6O15(OH)2·6H2O",
        crystal_system="正交晶系",
        card_number="01-075-1522",
        rir=1.2,
        peaks=[
            (7.30, 100.0), (12.14, 10.0), (16.64, 8.0), (20.64, 15.0),
            (23.84, 12.0), (26.54, 8.0), (32.14, 6.0), (35.04, 5.0),
            (39.54, 3.0), (45.24, 2.0), (50.14, 2.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="沸石 (Zeolite / Natrolite)",
        formula="Na2Al2Si3O10·2H2O",
        crystal_system="正交晶系",
        card_number="01-071-0962",
        rir=1.1,
        peaks=[
            (13.14, 100.0), (18.46, 20.0), (23.76, 15.0), (26.26, 25.0),
            (28.26, 10.0), (30.66, 8.0), (33.26, 12.0), (36.06, 6.0),
            (40.06, 5.0), (43.66, 4.0), (47.86, 3.0), (52.06, 2.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="钾长石 (Microcline)",
        formula="KAlSi3O8",
        crystal_system="三斜晶系",
        card_number="01-076-0804",
        rir=2.0,
        peaks=[
            (20.92, 35.0), (23.52, 100.0), (24.32, 12.0), (27.54, 18.0),
            (29.76, 12.0), (30.38, 22.0), (34.92, 8.0), (35.72, 6.0),
            (41.28, 5.0), (42.78, 4.0), (45.18, 5.0), (51.18, 3.0),
            (59.98, 3.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="斜长石 (Plagioclase)",
        formula="NaAlSi3O8-CaAl2Si2O8",
        crystal_system="三斜晶系",
        card_number="01-070-1733",
        rir=2.2,
        peaks=[
            (13.20, 15.0), (22.04, 25.0), (23.82, 30.0), (24.66, 100.0),
            (27.86, 10.0), (29.48, 18.0), (30.34, 12.0), (33.86, 8.0),
            (35.38, 6.0), (42.06, 4.0), (46.78, 3.0), (51.68, 2.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="白云母 (Muscovite)",
        formula="KAl2(AlSi3O10)(OH)2",
        crystal_system="单斜晶系",
        card_number="01-076-0666",
        rir=1.5,
        peaks=[
            (8.86, 100.0), (17.68, 15.0), (19.78, 18.0), (26.76, 20.0),
            (27.94, 10.0), (30.08, 8.0), (34.96, 10.0), (36.14, 6.0),
            (39.48, 4.0), (42.52, 3.0), (45.28, 4.0), (50.08, 3.0),
            (54.98, 2.0), (60.04, 2.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="黑云母 (Biotite)",
        formula="K(Mg,Fe)3(AlSi3O10)(OH)2",
        crystal_system="单斜晶系",
        card_number="01-074-1120",
        rir=1.6,
        peaks=[
            (8.98, 100.0), (17.86, 12.0), (19.96, 15.0), (26.88, 18.0),
            (28.06, 8.0), (30.24, 6.0), (34.88, 8.0), (36.06, 5.0),
            (39.68, 3.0), (42.68, 3.0), (45.48, 3.0), (50.28, 2.0),
            (55.18, 1.0), (60.14, 1.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="绿帘石 (Epidote)",
        formula="Ca2FeAl2(SiO4)(Si2O7)O(OH)",
        crystal_system="单斜晶系",
        card_number="01-072-0012",
        rir=2.2,
        peaks=[
            (15.32, 20.0), (23.14, 25.0), (26.14, 12.0), (28.64, 100.0),
            (30.64, 10.0), (32.14, 15.0), (34.64, 8.0), (36.14, 12.0),
            (38.64, 6.0), (41.14, 5.0), (44.64, 4.0), (47.14, 3.0),
            (50.64, 3.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="透辉石 (Diopside)",
        formula="CaMgSi2O6",
        crystal_system="单斜晶系",
        card_number="01-072-1086",
        rir=2.5,
        peaks=[
            (19.56, 15.0), (23.96, 20.0), (29.94, 100.0), (32.48, 10.0),
            (35.36, 12.0), (37.88, 8.0), (39.48, 5.0), (42.48, 6.0),
            (45.88, 4.0), (48.68, 3.0), (51.28, 3.0), (53.88, 2.0),
            (56.48, 2.0), (59.08, 2.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="普通辉石 (Augite)",
        formula="(Ca,Mg,Fe,Na)(Mg,Fe,Al)(Si,Al)2O6",
        crystal_system="单斜晶系",
        card_number="01-070-0954",
        rir=2.4,
        peaks=[
            (19.44, 12.0), (24.08, 18.0), (29.88, 100.0), (32.56, 10.0),
            (35.28, 12.0), (37.96, 8.0), (39.56, 5.0), (42.56, 5.0),
            (45.96, 3.0), (48.76, 3.0), (51.36, 2.0), (56.56, 2.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="角闪石 (Hornblende)",
        formula="Ca2(Mg,Fe)4Al(Si7Al)O22(OH)2",
        crystal_system="单斜晶系",
        card_number="01-072-0016",
        rir=1.8,
        peaks=[
            (10.52, 30.0), (18.86, 25.0), (21.64, 20.0), (27.84, 100.0),
            (30.14, 15.0), (33.24, 12.0), (35.34, 10.0), (38.14, 8.0),
            (42.04, 6.0), (45.34, 4.0), (49.24, 3.0), (53.14, 3.0),
            (57.04, 2.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="阳起石 (Actinolite)",
        formula="Ca2(Mg,Fe)5Si8O22(OH)2",
        crystal_system="单斜晶系",
        card_number="01-073-1916",
        rir=1.7,
        peaks=[
            (10.62, 35.0), (18.94, 20.0), (21.78, 18.0), (28.04, 100.0),
            (30.28, 12.0), (33.48, 10.0), (35.48, 8.0), (38.28, 6.0),
            (42.28, 5.0), (45.48, 3.0), (49.48, 3.0), (53.28, 2.0),
            (57.18, 2.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="蓝晶石 (Kyanite)",
        formula="Al2SiO5",
        crystal_system="三斜晶系",
        card_number="01-072-1524",
        rir=2.1,
        peaks=[
            (18.26, 20.0), (20.56, 25.0), (22.66, 100.0), (25.56, 10.0),
            (27.96, 15.0), (29.26, 8.0), (31.56, 12.0), (34.26, 6.0),
            (36.66, 8.0), (39.56, 5.0), (42.16, 4.0), (45.06, 3.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="红柱石 (Andalusite)",
        formula="Al2SiO5",
        crystal_system="正交晶系",
        card_number="01-073-1324",
        rir=2.0,
        peaks=[
            (15.56, 15.0), (20.36, 20.0), (22.36, 18.0), (25.06, 12.0),
            (27.36, 100.0), (30.36, 12.0), (33.26, 10.0), (35.46, 8.0),
            (38.66, 6.0), (41.36, 4.0), (44.26, 3.0), (47.16, 3.0),
            (50.06, 2.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="夕线石 (Sillimanite)",
        formula="Al2SiO5",
        crystal_system="正交晶系",
        card_number="01-073-1328",
        rir=2.0,
        peaks=[
            (14.86, 12.0), (20.06, 18.0), (22.06, 15.0), (25.86, 12.0),
            (28.36, 100.0), (31.26, 10.0), (34.36, 8.0), (36.96, 6.0),
            (39.26, 5.0), (42.06, 4.0), (45.46, 3.0), (49.06, 2.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="堇青石 (Cordierite)",
        formula="Mg2Al4Si5O18",
        crystal_system="正交晶系",
        card_number="01-072-1212",
        rir=1.4,
        peaks=[
            (10.52, 30.0), (12.62, 20.0), (18.02, 25.0), (21.72, 100.0),
            (26.32, 15.0), (28.42, 10.0), (30.32, 12.0), (33.02, 8.0),
            (36.22, 6.0), (40.02, 4.0), (43.42, 3.0), (46.82, 2.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="透闪石 (Tremolite)",
        formula="Ca2Mg5Si8O22(OH)2",
        crystal_system="单斜晶系",
        card_number="01-071-1204",
        rir=1.6,
        peaks=[
            (10.60, 40.0), (18.98, 25.0), (21.78, 20.0), (28.18, 100.0),
            (30.38, 15.0), (33.58, 12.0), (35.58, 10.0), (38.38, 8.0),
            (42.38, 6.0), (45.58, 4.0), (49.58, 3.0), (53.38, 3.0),
            (57.28, 2.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="滑石 (Talc)",
        formula="Mg3Si4O10(OH)2",
        crystal_system="单斜晶系",
        card_number="01-071-1180",
        rir=1.3,
        peaks=[
            (9.66, 100.0), (19.22, 20.0), (24.26, 15.0), (28.58, 12.0),
            (32.08, 8.0), (35.08, 10.0), (36.78, 6.0), (39.98, 4.0),
            (45.68, 3.0), (50.18, 2.0), (54.98, 2.0), (60.18, 1.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="叶蜡石 (Pyrophyllite)",
        formula="Al2Si4O10(OH)2",
        crystal_system="单斜晶系",
        card_number="01-070-1060",
        rir=1.4,
        peaks=[
            (9.48, 100.0), (18.78, 25.0), (21.38, 10.0), (24.98, 8.0),
            (28.28, 6.0), (32.08, 8.0), (35.18, 6.0), (37.98, 4.0),
            (42.08, 3.0), (45.78, 2.0), (50.28, 2.0), (55.08, 1.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="蛭石 (Vermiculite)",
        formula="(Mg,Fe,Al)3(Al,Si)4O10(OH)2·4H2O",
        crystal_system="单斜晶系",
        card_number="01-076-0848",
        rir=1.2,
        peaks=[
            (7.20, 100.0), (14.50, 10.0), (21.20, 15.0), (25.20, 12.0),
            (28.40, 8.0), (35.00, 8.0), (36.60, 5.0), (40.00, 3.0),
            (45.20, 3.0), (50.00, 2.0), (54.80, 2.0), (59.80, 1.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="坡缕石 (Palygorskite)",
        formula="Mg5Si8O20(OH)2·8H2O",
        crystal_system="单斜晶系",
        card_number="01-074-1284",
        rir=1.1,
        peaks=[
            (8.38, 100.0), (13.68, 20.0), (16.88, 15.0), (19.88, 12.0),
            (23.48, 10.0), (26.68, 8.0), (30.88, 6.0), (35.08, 5.0),
            (39.68, 3.0), (43.88, 2.0), (47.88, 2.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="海绿石 (Glauconite)",
        formula="(K,Na)(Fe,Al,Mg)2(Si,Al)4O10(OH)2",
        crystal_system="单斜晶系",
        card_number="01-073-1322",
        rir=1.3,
        peaks=[
            (8.80, 100.0), (17.60, 15.0), (19.70, 20.0), (26.60, 18.0),
            (29.90, 8.0), (35.00, 10.0), (36.50, 6.0), (40.00, 4.0),
            (45.20, 3.0), (50.10, 2.0), (55.00, 2.0), (60.00, 1.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="重晶石 (Barite)",
        formula="BaSO4",
        crystal_system="正交晶系",
        card_number="01-072-0544",
        rir=5.5,
        peaks=[
            (21.22, 12.0), (23.12, 20.0), (25.84, 100.0), (28.76, 8.0),
            (31.46, 5.0), (32.86, 10.0), (39.28, 5.0), (42.58, 6.0),
            (45.36, 3.0), (48.86, 4.0), (53.26, 2.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="天青石 (Celestite)",
        formula="SrSO4",
        crystal_system="正交晶系",
        card_number="01-072-0512",
        rir=4.5,
        peaks=[
            (21.64, 15.0), (23.54, 25.0), (26.24, 100.0), (29.14, 10.0),
            (31.84, 6.0), (33.24, 12.0), (39.64, 6.0), (42.94, 8.0),
            (45.74, 4.0), (49.24, 5.0), (53.64, 3.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="磷灰石 (Apatite)",
        formula="Ca5(PO4)3(F,Cl,OH)",
        crystal_system="六方晶系",
        card_number="01-072-0712",
        rir=3.0,
        peaks=[
            (21.70, 15.0), (22.90, 20.0), (25.80, 10.0), (28.90, 100.0),
            (31.90, 25.0), (32.90, 20.0), (34.10, 15.0), (36.30, 8.0),
            (39.50, 10.0), (41.60, 5.0), (44.10, 3.0), (46.70, 5.0),
            (49.40, 4.0), (52.10, 3.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="石盐 (Halite)",
        formula="NaCl",
        crystal_system="等轴晶系",
        card_number="01-071-1338",
        rir=2.4,
        peaks=[
            (27.32, 100.0), (31.70, 15.0), (45.44, 30.0), (53.86, 10.0),
            (56.48, 15.0), (66.22, 5.0), (75.30, 4.0), (84.26, 2.0),
        ]
    ))
    
    cards.append(PDFCard(
        name="钾盐 (Sylvite)",
        formula="KCl",
        crystal_system="等轴晶系",
        card_number="01-071-1448",
        rir=2.8,
        peaks=[
            (24.00, 100.0), (28.38, 8.0), (40.50, 25.0), (48.14, 8.0),
            (50.50, 12.0), (60.00, 4.0), (68.68, 3.0), (76.98, 2.0),
        ]
    ))
    
    return cards


_pdf_database = None


def get_pdf_database() -> List[PDFCard]:
    """获取PDF卡片数据库"""
    global _pdf_database
    if _pdf_database is None:
        _pdf_database = _build_pdf_database()
    return _pdf_database


def search_pdf_by_name(name: str) -> List[PDFCard]:
    """按名称搜索PDF卡片"""
    db = get_pdf_database()
    name_lower = name.lower()
    return [c for c in db if name_lower in c.name.lower() or name_lower in c.formula.lower()]


def get_pdf_card_by_name(name: str) -> PDFCard:
    """根据名称获取PDF卡片（精确匹配）"""
    db = get_pdf_database()
    for card in db:
        if card.name == name:
            return card
    return None
