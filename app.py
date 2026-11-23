"""
Tytuł pracy inżynierskiej: Wirtualny asystent trenera, projekt aplikacji do symulacji parametrów życiowych sportowca
Autor: Wiktor Kubacki
Uczelnia: WSB Merito
Opis: Moduł symulacji telemetrii wysiłkowej oparty na modelach fizjologicznych oraz klasyfikatorze ML
"""

import json
import random
import time
from typing import Tuple
import plotly.graph_objects as go
import streamlit as st
from sklearn.tree import DecisionTreeClassifier


# --- TRENING MODELU UCZENIA MASZYNOWEGO (SCIKIT-LEARN) ---
@st.cache_resource
def stworz_i_wytrenuj_model_ai() -> DecisionTreeClassifier:
    """
    Generuje przykładowe dane treningowe i uczy model Drzewa Decyzyjnego
    klasyfikowania stanu sportowca na trzy komendy: ZWOLNIJ, PRZYSPIESZ, UTRZYMAJ TEMPO.
    """
    X_dane = []      # Lista na parametry wejściowe: [procent_HRR, BMI, czas_sesji]
    y_decyzje = []   # Lista na właściwe decyzje trenera (etykiety klas)

    for _ in range(1500):
        losowe_hrr = random.uniform(0.30, 1.00)
        losowe_bmi = random.uniform(18.0, 35.0)
        losowy_czas = random.randint(5, 120)

        if losowe_bmi >= 28.0:
            dolny_prog, gorny_prog = 0.60, 0.80
        elif losowy_czas <= 20:
            dolny_prog, gorny_prog = 0.75, 0.90
        else:
            dolny_prog, gorny_prog = 0.60, 0.75

        if losowe_hrr > gorny_prog:
            decyzja = "ZWOLNIJ"
        elif losowe_hrr < dolny_prog:
            decyzja = "PRZYSPIESZ"
        else:
            decyzja = "UTRZYMAJ TEMPO"

        X_dane.append([losowe_hrr, losowe_bmi, losowy_czas])
        y_decyzje.append(decyzja)

    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_dane, y_decyzje)
    return model


class WirtualnySportowiec:
    """
    Model symulacyjny sportowca odwzorowujący reakcję układu krążenia,
    kumulację zmęczenia oraz wydatek energetyczny w czasie rzeczywistym.
    """

    def __init__(self, wiek: int, waga: float, wzrost: float, plec: str, tetno_spoczynkowe: float) -> None:
        self.czas_treningu_s: int = 0
        self.czas_calkowity_s: int = 0
        self.zmeczenie_kumulatywne: float = 0.0
        self.spalone_kcal: float = 0.0
        self.aktualne_tetno: float = float(tetno_spoczynkowe)

        self.aktualizuj_parametry(wiek, waga, wzrost, plec, tetno_spoczynkowe)

        # Stochastyczne cechy osobnicze (losowane dla unikalności sesji treningowej)
        self.dynamika_wzrostu: float = random.uniform(0.7, 1.3) * (1.0 + self.kondycja)
        self.dynamika_spadku: float = random.uniform(0.7, 1.3) * (2.0 - self.kondycja)
        self.niestabilnosc_tempa: float = random.uniform(0.4, 1.8)
        self.trend_szumu: float = random.uniform(-1.0, 1.0)

    def aktualizuj_parametry(self, wiek: int, waga: float, wzrost: float, plec: str, tetno_spoczynkowe: float) -> None:
        """Dynamiczna aktualizacja parametrów biometrycznych zawodnika w locie."""
        self.wiek = wiek
        self.waga = waga
        self.wzrost = wzrost
        self.plec = plec
        self.tetno_spoczynkowe = float(tetno_spoczynkowe)

        # Wyznaczenie tętna maksymalnego wg Sally Edwards
        if plec == "Mężczyzna":
            self.max_tetno = 214 - (0.8 * wiek) - (0.11 * waga)
        else:
            self.max_tetno = 210 - (0.5 * wiek) - (0.11 * waga)

        self.max_tetno = round(self.max_tetno)
        self.hrr = self.max_tetno - self.tetno_spoczynkowe

        wzrost_m = wzrost / 100.0
        self.bmi = waga / (wzrost_m ** 2)

        wspolczynnik_kondycji = (self.tetno_spoczynkowe - 40.0) / 40.0
        self.kondycja = max(0.0, min(1.0, wspolczynnik_kondycji))

    def aktualizuj_stan(self, poziom_wysilku: int, tryb_pauzy: bool = False) -> int:
        """Wylicza kolejny krok czasowy (takt 1 sekundy) stanu fizjologicznego."""
        self.czas_calkowity_s += 1
        if not tryb_pauzy:
            self.czas_treningu_s += 1

        if tryb_pauzy:
            docelowe_hr = self.tetno_spoczynkowe
        else:
            if poziom_wysilku == 0:
                docelowe_hr = self.tetno_spoczynkowe
            elif poziom_wysilku == 1:
                docelowe_hr = (self.hrr * 0.60) + self.tetno_spoczynkowe
            elif poziom_wysilku == 2:
                docelowe_hr = (self.hrr * 0.70) + self.tetno_spoczynkowe
            elif poziom_wysilku == 3:
                docelowe_hr = (self.hrr * 0.80) + self.tetno_spoczynkowe
            else:
                docelowe_hr = (self.hrr * 0.95) + self.tetno_spoczynkowe

        roznica = docelowe_hr - self.aktualne_tetno

        if roznica > 0:
            wspolczynnik_adaptacji = 0.06 * self.dynamika_wzrostu
        else:
            wspolczynnik_adaptacji = 0.008 * self.dynamika_spadku

        mnoznik_zmeczenia = 1.0 + (self.kondycja * 0.5)
        if self.bmi > 25.0:
            mnoznik_zmeczenia += 0.3

        if poziom_wysilku >= 3 and not tryb_pauzy:
            self.zmeczenie_kumulatywne += 0.04 * mnoznik_zmeczenia
        elif poziom_wysilku <= 1 or tryb_pauzy:
            if self.zmeczenie_kumulatywne > 0:
                self.zmeczenie_kumulatywne -= 0.02 / mnoznik_zmeczenia

        self.trend_szumu = (self.trend_szumu * 0.85) + (random.uniform(-1.0, 1.0) * self.niestabilnosc_tempa)
        przyrost = roznica * wspolczynnik_adaptacji

        if tryb_pauzy:
            wspolczynnik_pauzy = 0.008 * self.dynamika_spadku
            self.aktualne_tetno += (docelowe_hr - self.aktualne_tetno) * wspolczynnik_pauzy + (self.trend_szumu * 0.2)
        else:
            wplyw_dryfu = (self.zmeczenie_kumulatywne * 0.04 * (self.niestabilnosc_tempa * 0.8)) if poziom_wysilku > 0 else 0
            self.aktualne_tetno += przyrost + self.trend_szumu + wplyw_dryfu

        if self.aktualne_tetno > self.max_tetno:
            self.aktualne_tetno = self.max_tetno
        if self.aktualne_tetno < self.tetno_spoczynkowe:
            self.aktualne_tetno = self.tetno_spoczynkowe

        hr = self.aktualne_tetno
        if self.plec == "Mężczyzna":
            kcal_per_min = (-55.0969 + (0.6309 * hr) + (0.09017 * self.waga * 2.204) + (0.2017 * self.wiek)) / 4.184
        else:
            kcal_per_min = (-20.4022 + (0.4472 * hr) - (0.05729 * self.waga * 2.204) + (0.074 * self.wiek)) / 4.184

        self.spalone_kcal += max(0, kcal_per_min / 60.0)
        return int(self.aktualne_tetno)


class AsystentAI:
    """
    Moduł decyzyjny wykorzystujący model uczenia maszynowego (scikit-learn)
    do klasyfikacji danych telemetrycznych i generowania decyzji treningowych.
    """

    def __init__(self, target_duration_min: int, bmi: float, hrr: float, tetno_spoczynkowe: float) -> None:
        self.model_ml = stworz_i_wytrenuj_model_ai()
        self.aktualizuj_parametry(target_duration_min, bmi, hrr, tetno_spoczynkowe)

    def aktualizuj_parametry(self, target_duration_min: int, bmi: float, hrr: float, tetno_spoczynkowe: float) -> None:
        self.target_duration = target_duration_min
        self.bmi = bmi
        self.hrr = hrr
        self.tetno_spoczynkowe = tetno_spoczynkowe

    def generuj_rekomendacje(self, hr: float, elapsed_s: int, tryb_pauzy: bool = False) -> Tuple[str, str, str]:
        """Zwraca krotkę: (komenda_glowna, opis_kontekstu, kolor_statusu)."""
        if tryb_pauzy:
            return "AKTYWNA REGENERACJA", "Wstrzymano obciążenie. Trwa faza schładzania organizmu (Cool-down).", "#facc15"

        elapsed_min = elapsed_s / 60.0
        progress = (elapsed_min / self.target_duration) * 100

        if progress > 100:
            return "CZAS PLANOWY OSIĄGNIĘTY", "Zakończ sesję lub przejdź do spokojnego marszu schładzającego.", "#38bdf8"

        if self.bmi >= 28.0:
            kontekst = "Profil: Ochrona stawów (BMI >= 28,0) | Strefa docelowa: 60-80% HRR"
        elif self.target_duration <= 20:
            kontekst = "Profil: Sesja interwałowa (HIIT) | Strefa docelowa: 75-90% HRR"
        else:
            kontekst = "Profil: Trening wytrzymałościowy | Strefa docelowa: 60-75% HRR"

        aktualne_hrr = (hr - self.tetno_spoczynkowe) / self.hrr if self.hrr > 0 else 0.0
        biezace_dane = [[aktualne_hrr, self.bmi, self.target_duration]]
        decyzja_ai = self.model_ml.predict(biezace_dane)[0]

        if decyzja_ai == "ZWOLNIJ":
            return "ZWOLNIJ", f"Przekroczono górny próg tętna. {kontekst}", "#ef4444"
        elif decyzja_ai == "PRZYSPIESZ":
            return "PRZYSPIESZ", f"Intensywność poniżej założonego celu. {kontekst}", "#f59e0b"
        else:
            return "UTRZYMAJ TEMPO", f"Parametry krążeniowe w optymalnym oknie. {kontekst}", "#10b981"


# --- KONFIGURACJA STRONY I STYLÓW INTERFEJSU ---
st.set_page_config(layout="wide", page_title="Wirtualny Asystent Trenera - Symulator")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

        body, p, h1, h2, h3, h4, label, input, button,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricLabel"],
        [data-testid="stCaptionContainer"] {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
        }

        .stApp {
            background-color: #0e0f12;
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 1.5rem;
        }

        [data-testid="stSidebar"] {
            background-color: #18191c !important;
            border-right: 1px solid #2b2d33;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #18191c !important;
            border: 1px solid #2b2d33 !important;
            border-radius: 10px !important;
        }

        .hr-card {
            background-color: #18191c;
            border: 1px solid #2b2d33;
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 12px;
        }
        .hr-label { font-size: 11px; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
        .hr-value { font-size: 42px; font-weight: 700; color: #e5e7eb; line-height: 1.1; margin: 4px 0; }
        .hr-sub { font-size: 13px; color: #d1d5db; }

        .ai-box {
            background-color: #18191c;
            border: 1px solid #2b2d33;
            border-radius: 10px;
            padding: 14px 18px;
            margin-bottom: 14px;
        }
        .ai-cmd { font-size: 20px; font-weight: 700; margin: 2px 0 4px 0; }
        .ai-desc { font-size: 12px; color: #9ca3af; line-height: 1.4; }
    </style>
""", unsafe_allow_html=True)

# Inicjalizacja stanu sesji
if 'live' not in st.session_state: st.session_state.live = False
if 'pauza' not in st.session_state: st.session_state.pauza = False
if 'poziom_wysilku' not in st.session_state: st.session_state.poziom_wysilku = 0
if 'sesja_zakonczona' not in st.session_state: st.session_state.sesja_zakonczona = False
if 'hist_hr' not in st.session_state: st.session_state.hist_hr = []
if 'hist_t' not in st.session_state: st.session_state.hist_t = []
if 'ostatni_import' not in st.session_state: st.session_state.ostatni_import = None

# Domyślne wartości profilu zawodnika w pamięci sesji
if 'cfg_plec' not in st.session_state: st.session_state.cfg_plec = "Mężczyzna"
if 'cfg_wiek' not in st.session_state: st.session_state.cfg_wiek = 25
if 'cfg_waga' not in st.session_state: st.session_state.cfg_waga = 78.0
if 'cfg_wzrost' not in st.session_state: st.session_state.cfg_wzrost = 180.0
if 'cfg_hr_rest' not in st.session_state: st.session_state.cfg_hr_rest = 55
if 'cfg_plan_min' not in st.session_state: st.session_state.cfg_plan_min = 30

# --- PANEL BOCZNY: IMPORT, PROFIL BIOMETRYCZNY I EKSPORT ---
st.sidebar.markdown("### Profil Zawodnika")

with st.sidebar.expander("Wczytaj profil z pliku (.json)"):
    wgrany_plik = st.file_uploader("Wybierz plik JSON", type=["json"], label_visibility="collapsed")
    if wgrany_plik is not None:
        id_pliku = f"{wgrany_plik.name}_{wgrany_plik.size}"
        if st.session_state.ostatni_import != id_pliku:
            try:
                dane_profilu = json.load(wgrany_plik)
                if dane_profilu.get("plec") in ["Mężczyzna", "Kobieta"]:
                    st.session_state.cfg_plec = dane_profilu["plec"]
                st.session_state.cfg_wiek = max(15, min(90, int(dane_profilu.get("wiek", 25))))
                st.session_state.cfg_waga = max(30.0, min(250.0, float(dane_profilu.get("waga", 78.0))))
                st.session_state.cfg_wzrost = max(100.0, min(250.0, float(dane_profilu.get("wzrost", 180.0))))
                st.session_state.cfg_hr_rest = max(35, min(110, int(dane_profilu.get("hr_rest", 55))))
                st.session_state.cfg_plan_min = max(5, min(120, int(dane_profilu.get("plan_min", 30))))
                st.session_state.ostatni_import = id_pliku
                st.success("Wczytano ustawienia zawodnika.")
            except Exception:
                st.error("Nieprawidłowy format pliku JSON.")
    else:
        st.session_state.ostatni_import = None

plec = st.sidebar.selectbox("Płeć", ["Mężczyzna", "Kobieta"], key="cfg_plec")
wiek = st.sidebar.number_input("Wiek (lata)", min_value=15, max_value=90, step=1, key="cfg_wiek")
waga = st.sidebar.number_input("Masa ciała (kg)", min_value=30.0, max_value=250.0, step=0.5, key="cfg_waga")
wzrost = st.sidebar.number_input("Wzrost (cm)", min_value=100.0, max_value=250.0, step=0.5, key="cfg_wzrost")
hr_rest = st.sidebar.number_input("Tętno spoczynkowe (BPM)", min_value=35, max_value=110, step=1, key="cfg_hr_rest")
plan_min = st.sidebar.slider("Planowany czas sesji (min)", min_value=5, max_value=120, key="cfg_plan_min")

# Instancjonowanie obiektów
if 'sportowiec' not in st.session_state:
    st.session_state.sportowiec = WirtualnySportowiec(wiek, waga, wzrost, plec, hr_rest)
if 'asystent' not in st.session_state:
    st.session_state.asystent = AsystentAI(
        plan_min,
        st.session_state.sportowiec.bmi,
        st.session_state.sportowiec.hrr,
        st.session_state.sportowiec.tetno_spoczynkowe
    )

s = st.session_state.sportowiec
ai = st.session_state.asystent

s.aktualizuj_parametry(wiek, waga, wzrost, plec, hr_rest)
ai.aktualizuj_parametry(plan_min, s.bmi, s.hrr, s.tetno_spoczynkowe)

st.sidebar.divider()
st.sidebar.markdown("### Parametry Wyliczone")
st.sidebar.write(f"**Tętno maksymalne:** {s.max_tetno} BPM")
st.sidebar.write(f"**Rezerwa tętna (HRR):** {int(s.hrr)} BPM")
st.sidebar.write(f"**Wskaźnik BMI:** {s.bmi:.1f} kg/m²")

# Przygotowanie pliku JSON do eksportu aktualnego profilu
profil_do_zapisu = json.dumps(
    {
        "plec": plec,
        "wiek": wiek,
        "waga": waga,
        "wzrost": wzrost,
        "hr_rest": hr_rest,
        "plan_min": plan_min
    },
    ensure_ascii=False,
    indent=4
)
st.sidebar.download_button(
    label="Zapisz profil (.json)",
    data=profil_do_zapisu,
    file_name="profil_zawodnika.json",
    mime="application/json",
    use_container_width=True
)

# --- AKTUALIZACJA KROKU SYMULACJI ---
if st.session_state.live:
    hr_now = s.aktualizuj_stan(st.session_state.poziom_wysilku, tryb_pauzy=st.session_state.pauza)
    st.session_state.hist_hr.append(hr_now)
    st.session_state.hist_t.append(s.czas_calkowity_s)
else:
    hr_now = st.session_state.hist_hr[-1] if st.session_state.hist_hr else int(hr_rest)

avg_hr = int(sum(st.session_state.hist_hr) / len(st.session_state.hist_hr)) if st.session_state.hist_hr else int(hr_rest)
max_hr_achieved = max(st.session_state.hist_hr) if st.session_state.hist_hr else int(hr_rest)
trimp = int((s.czas_treningu_s / 60) * (avg_hr / s.max_tetno) * 1.5) if s.max_tetno > 0 else 0
procent_hrr_now = max(0, int(((hr_now - s.tetno_spoczynkowe) / s.hrr) * 100)) if s.hrr > 0 else 0

# --- NAGŁÓWEK GŁÓWNY I KOLOROWY PASEK POSTĘPU ---
head_left, head_right = st.columns([3, 1])
with head_left:
    st.markdown("## Wirtualny Asystent Trenera")
with head_right:
    if st.session_state.live and not st.session_state.pauza:
        status_txt = "STATUS: REJESTRACJA AKTYWNA"
    elif st.session_state.pauza:
        status_txt = "STATUS: PAUZA (SCHŁADZANIE)"
    elif st.session_state.sesja_zakonczona:
        status_txt = "STATUS: SESJA ZAKOŃCZONA"
    else:
        status_txt = "STATUS: GOTOWOŚĆ DO STARTU"
    st.caption(f"**{status_txt}**")

procent_treningu = min(100.0, (s.czas_treningu_s / (plan_min * 60)) * 100) if plan_min > 0 else 0.0

if st.session_state.sesja_zakonczona:
    kolor_paska = "#ef4444"   # Czerwony po kliknięciu STOP
elif st.session_state.pauza:
    kolor_paska = "#facc15"   # Żółty podczas PAUZY
else:
    kolor_paska = "#00e5ff"   # Cyjan podczas pracy / gotowości

st.markdown(f"""
    <div style="width: 100%; background-color: #1f2228; border-radius: 6px; height: 10px; overflow: hidden; margin-bottom: 6px;">
        <div style="width: {procent_treningu}%; background-color: {kolor_paska}; height: 100%; transition: width 0.3s ease;"></div>
    </div>
    <div style="font-size: 13px; font-weight: 600; color: {kolor_paska}; margin-bottom: 14px;">
        Postęp zaplanowanego treningu: {procent_treningu:.1f}%
    </div>
""", unsafe_allow_html=True)

# --- GŁÓWNY UKŁAD DWUKOLUMNOWY ---
col_wykres, col_kokpit = st.columns([2.2, 1.0], gap="large")

with col_kokpit:
    # 1. Karta bieżącego tętna
    st.markdown(f"""
        <div class="hr-card">
            <div class="hr-label">Aktualna częstość skurczów serca</div>
            <div class="hr-value">{hr_now} <span style="font-size: 18px; color: #9ca3af;">BPM</span></div>
            <div class="hr-sub">Obciążenie układu krążenia: <b>{procent_hrr_now}% HRR</b> &nbsp;|&nbsp; Czas: <b>{s.czas_treningu_s // 60}:{s.czas_treningu_s % 60:02d} / {plan_min}:00</b></div>
        </div>
    """, unsafe_allow_html=True)

    # 2. Karta decyzji modelu scikit-learn
    if st.session_state.live:
        komenda, opis_ml, kolor_ramki = ai.generuj_rekomendacje(hr_now, s.czas_treningu_s, tryb_pauzy=st.session_state.pauza)
    else:
        komenda = "OCZEKIWANIE NA START"
        opis_ml = "Dostosuj parametry zawodnika w panelu bocznym i uruchom sesję, aby aktywować klasyfikator DecisionTree."
        kolor_ramki = "#9ca3af"

    st.markdown(f"""
        <div class="ai-box" style="border-left: 5px solid {kolor_ramki};">
            <div class="hr-label">Rekomendacja modułu ML (scikit-learn)</div>
            <div class="ai-cmd" style="color: {kolor_ramki};">{komenda}</div>
            <div class="ai-desc">{opis_ml}</div>
        </div>
    """, unsafe_allow_html=True)

    # 3. Panel sterowania sesją i interwencji trenerskiej
    with st.container(border=True):
        st.markdown("**Sterowanie sesją**")
        c_start, c_pauza, c_stop = st.columns(3)
        if c_start.button("START", use_container_width=True, type="primary"):
            st.session_state.live = True
            st.session_state.pauza = False
            st.session_state.sesja_zakonczona = False
            st.session_state.poziom_wysilku = 1
            st.session_state.hist_hr = []
            st.session_state.hist_t = []
            st.session_state.sportowiec = WirtualnySportowiec(wiek, waga, wzrost, plec, hr_rest)
            s_obj = st.session_state.sportowiec
            st.session_state.asystent = AsystentAI(plan_min, s_obj.bmi, s_obj.hrr, s_obj.tetno_spoczynkowe)
            st.rerun()

        etykieta_pauza = "WZNÓW" if st.session_state.pauza else "PAUZA"
        if c_pauza.button(etykieta_pauza, use_container_width=True):
            if st.session_state.live:
                st.session_state.pauza = not st.session_state.pauza
                st.rerun()

        if c_stop.button("STOP", use_container_width=True):
            if st.session_state.live:
                st.session_state.live = False
                st.session_state.pauza = False
                st.session_state.sesja_zakonczona = True
                st.rerun()

        if st.session_state.sesja_zakonczona or len(st.session_state.hist_hr) > 0:
            if st.button("Resetuj sesję", use_container_width=True):
                st.session_state.live = False
                st.session_state.pauza = False
                st.session_state.sesja_zakonczona = False
                st.session_state.poziom_wysilku = 0
                st.session_state.hist_hr = []
                st.session_state.hist_t = []
                if 'sportowiec' in st.session_state:
                    del st.session_state.sportowiec
                st.rerun()

        if st.session_state.live and not st.session_state.pauza:
            st.divider()
            st.markdown("**Interwencja trenerska (korekta tempa)**")
            b1, b2, b3 = st.columns(3)
            if b1.button("- Zwolnij", use_container_width=True):
                st.session_state.poziom_wysilku = max(0, st.session_state.poziom_wysilku - 1)
            if b2.button("Utrzymaj", use_container_width=True):
                pass
            if b3.button("+ Przyspiesz", use_container_width=True):
                st.session_state.poziom_wysilku = min(4, st.session_state.poziom_wysilku + 1)

            etykiety_wysilku = ["0: Spoczynek", "1: Lekki trucht", "2: Bieg optymalny", "3: Mocny bieg", "4: Maksymalny sprint"]
            st.caption(f"Zadany poziom: **{etykiety_wysilku[st.session_state.poziom_wysilku]}**")

    # 4. Siatka 2x2 statystyk pomocniczych
    with st.container(border=True):
        r1c1, r1c2 = st.columns(2)
        r1c1.metric("Tętno średnie", f"{avg_hr} BPM")
        r1c2.metric("Wydatek energii", f"{int(s.spalone_kcal)} kcal")

        r2c1, r2c2 = st.columns(2)
        r2c1.metric("Obciążenie TRIMP", f"{trimp} pkt")
        r2c2.metric("Zmęczenie kumul.", f"{s.zmeczenie_kumulatywne:.2f}")

with col_wykres:
    # --- WIZUALIZACJA PLOTLY ---
    fig = go.Figure()
    teksty_etykiet = [""] * len(st.session_state.hist_hr)
    if teksty_etykiet:
        teksty_etykiet[-1] = f"  {hr_now} BPM"

    fig.add_trace(go.Scatter(
        x=st.session_state.hist_t,
        y=st.session_state.hist_hr,
        mode='lines+text',
        name='Tętno zawodnika',
        line=dict(color='#9ca3af', width=3),
        text=teksty_etykiet,
        textposition="middle right",
        textfont=dict(color="#9ca3af", size=13, family="Plus Jakarta Sans, sans-serif"),
        cliponaxis=False,
        hovertemplate="<b>Czas:</b> %{x}<br><b>Tętno:</b> %{y}<extra></extra>"
    ))

    # Wyznaczanie stref Karvonena
    k_50 = (s.hrr * 0.50) + s.tetno_spoczynkowe
    k_60 = (s.hrr * 0.60) + s.tetno_spoczynkowe
    k_70 = (s.hrr * 0.70) + s.tetno_spoczynkowe
    k_80 = (s.hrr * 0.80) + s.tetno_spoczynkowe
    k_90 = (s.hrr * 0.90) + s.tetno_spoczynkowe

    zones = [
        (k_50, k_60, "Strefa 1 (50-60% HRR): Regeneracja", "#64748b"),
        (k_60, k_70, "Strefa 2 (60-70% HRR): Spalanie tłuszczu", "#0284c7"),
        (k_70, k_80, "Strefa 3 (70-80% HRR): Aerobowa", "#16a34a"),
        (k_80, k_90, "Strefa 4 (80-90% HRR): Anaerobowa", "#d97706"),
        (k_90, s.max_tetno, "Strefa 5 (90-100% HRR): Maksymalna / VO2 Max", "#dc2626")
    ]

    for low, high, label, color in zones:
        if high > low:
            fig.add_hrect(
                y0=low, y1=high,
                fillcolor=color, opacity=0.14,
                layer="below", line_width=0,
                annotation_text=label,
                annotation_position="top left",
                annotation_font=dict(size=11, color="rgba(255,255,255,0.75)", family="Plus Jakarta Sans, sans-serif")
            )

    for granica in [k_60, k_70, k_80, k_90]:
        fig.add_hline(
            y=granica,
            line_dash="dot",
            line_color="rgba(255, 255, 255, 0.25)",
            line_width=1,
            layer="below"
        )

    fig.update_layout(
        height=540,
        margin=dict(l=60, r=85, t=20, b=50),
        font=dict(family="Plus Jakarta Sans, sans-serif"),
        xaxis=dict(
            title="Czas rejestracji",
            ticksuffix=" sek",
            showticklabels=True,
            tickfont=dict(color="#9ca3af", size=12),
            title_font=dict(color="#9ca3af", size=12),
            gridcolor="rgba(255, 255, 255, 0.08)"
        ),
        yaxis=dict(
            range=[40, max(210, s.max_tetno + 10)],
            title="Częstość skurczów serca",
            ticksuffix=" BPM",
            showticklabels=True,
            tickfont=dict(color="#9ca3af", size=12),
            title_font=dict(color="#9ca3af", size=12),
            gridcolor="rgba(255, 255, 255, 0.08)"
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#131519',
        hoverlabel=dict(
            bgcolor="#18191c",
            font_size=13,
            font_family="Plus Jakarta Sans, sans-serif"
        )
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # --- KOMPAKTOWA ARCHIWIZACJA I EKSPORT RAPORTU PO ZAKOŃCZENIU SESJI ---
    if st.session_state.sesja_zakonczona:
        exp_col, _ = st.columns([1.15, 1.05])
        with exp_col:
            with st.container(border=True):
                st.markdown("**Archiwizacja i eksport sesji treningowej**")
                with st.form(key="formularz_pobierania"):
                    f_col1, f_col2 = st.columns([1.3, 1.0])
                    nazwa_treningu = f_col1.text_input("Nazwa pliku raportu:", value="Raport_Treningowy", label_visibility="collapsed")
                    zatwierdzono = f_col2.form_submit_button("Generuj raport HTML", use_container_width=True)

                if zatwierdzono:
                    czysta_nazwa = nazwa_treningu.strip().replace(" ", "_") or "Raport_Treningowy"
                    plik_nazwa = f"{czysta_nazwa}.html"
                    wykres_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

                    raport_html_content = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <meta charset="utf-8">
                        <title>Raport Telemetryczny - {czysta_nazwa}</title>
                        <style>
                            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background-color: #0e0f12; color: #e5e7eb; }}
                            .container {{ max-width: 1200px; margin: 0 auto; }}
                            .header {{ border-bottom: 1px solid #2b2d33; padding-bottom: 20px; margin-bottom: 30px; }}
                            .header h1 {{ margin: 0; color: #e5e7eb; font-size: 26px; }}
                            .header p {{ color: #9ca3af; margin: 6px 0 0 0; }}
                            .stats-grid {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 15px; margin-bottom: 30px; }}
                            .stat-card {{ background-color: #18191c; border: 1px solid #2b2d33; padding: 15px; border-radius: 8px; text-align: center; }}
                            .stat-label {{ font-size: 11px; color: #9ca3af; text-transform: uppercase; margin-bottom: 8px; }}
                            .stat-value {{ font-size: 22px; font-weight: bold; color: #ffffff; margin: 0; }}
                            .chart-container {{ background-color: #18191c; border: 1px solid #2b2d33; border-radius: 8px; padding: 20px; }}
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <div class="header">
                                <h1>Raport z Telemetrii Wysiłkowej: {czysta_nazwa}</h1>
                                <p>Wygenerowano przez Wirtualny Asystent Trenera (Model Karvonena, Sally Edwards &amp; DecisionTree ML)</p>
                            </div>
                            <div class="stats-grid">
                                <div class="stat-card"><div class="stat-label">Czas Sesji</div><div class="stat-value">{s.czas_treningu_s // 60}:{s.czas_treningu_s % 60:02d}</div></div>
                                <div class="stat-card"><div class="stat-label">Tętno AVG</div><div class="stat-value">{avg_hr} BPM</div></div>
                                <div class="stat-card"><div class="stat-label">Tętno MAX</div><div class="stat-value">{max_hr_achieved} BPM</div></div>
                                <div class="stat-card"><div class="stat-label">Spalone Kalorie</div><div class="stat-value">{int(s.spalone_kcal)} kcal</div></div>
                                <div class="stat-card"><div class="stat-label">Obciążenie TRIMP</div><div class="stat-value">{trimp}</div></div>
                                <div class="stat-card"><div class="stat-label">Wskaźnik BMI</div><div class="stat-value">{s.bmi:.1f}</div></div>
                            </div>
                            <div class="chart-container">{wykres_html}</div>
                        </div>
                    </body>
                    </html>
                    """
                    st.session_state.pobierz_dane = raport_html_content
                    st.session_state.pobierz_nazwa = plik_nazwa

                if 'pobierz_dane' in st.session_state and st.session_state.pobierz_dane:
                    st.download_button(
                        label=f"Pobierz plik: {st.session_state.pobierz_nazwa}",
                        data=st.session_state.pobierz_dane,
                        file_name=st.session_state.pobierz_nazwa,
                        mime="text/html",
                        use_container_width=True
                    )

if st.session_state.live:
    time.sleep(1)
    st.rerun()
