"""
Tytuł pracy inżynierskiej: Wirtualny asystent trenera, projekt aplikacji do symulacji parametrów życiowych sportowca
Autor: Wiktor Kubacki
Uczelnia: WSB Merito
Opis: Moduł symulacji telemetrii wysiłkowej oparty na wielu modelach fizjologicznych
"""

import random
import time
from typing import Tuple
import plotly.graph_objects as go
import streamlit as st
from sklearn.tree import DecisionTreeClassifier


# Trening modelu uczenia maszynowego SCIKIT-LEARN
@st.cache_resource
def stworz_i_wytrenuj_model_ai() -> DecisionTreeClassifier:
    """
    Generuje przykładowe dane treningowe i uczy model Drzewa Decyzyjnego
    klasyfikowania stanu sportowca na trzy komendy: ZWOLNIJ, PRZYSPIESZ, UTRZYMAJ TEMPO.
    """
    X_dane = []      # Lista na parametry wejściowe: procent_HRR + BMI + czas_sesji
    y_decyzje = []   # Lista na właściwe decyzje trenera - etykiety klas

    # Generowanie 1500 przykładowych sytuacji treningowych do nauki modelu
    for _ in range(1500):
        losowe_hrr = random.uniform(0.30, 1.00)   # Obciążenie tętna od 30% do 100%
        losowe_bmi = random.uniform(18.0, 35.0)   # Losowe BMI
        losowy_czas = random.randint(5, 120)      # Losowy planowany czas w min

        # Wyznaczenie prawidłowych progów dla wylosowanego przykładu
        if losowe_bmi >= 28.0:
            dolny_prog, gorny_prog = 0.60, 0.80
        elif losowy_czas <= 20:
            dolny_prog, gorny_prog = 0.75, 0.90
        else:
            dolny_prog, gorny_prog = 0.60, 0.75

        # Przypisanie poprawnej decyzji (klasy) do wylosowanych parametrów
        if losowe_hrr > gorny_prog:
            decyzja = "ZWOLNIJ"
        elif losowe_hrr < dolny_prog:
            decyzja = "PRZYSPIESZ"
        else:
            decyzja = "UTRZYMAJ TEMPO"

        X_dane.append([losowe_hrr, losowe_bmi, losowy_czas])
        y_decyzje.append(decyzja)

    # Utworzenie + wytrenowanie klasyfikatora z scikit-learn
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
        self.zmeczenie_kumulatywne: float = 0.0
        self.spalone_kcal: float = 0.0
        self.aktualne_tetno: float = float(tetno_spoczynkowe)

        # Inicjalizacja biometrii bazowej
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

        # Obliczenie rezerwy tętna (HRR) wg Karvonena
        self.hrr = self.max_tetno - self.tetno_spoczynkowe

        wzrost_m = wzrost / 100.0
        self.bmi = waga / (wzrost_m ** 2)

        # Wskaźnik kondycji fizjologicznej bazujący na spoczynkowym tętnie
        wspolczynnik_kondycji = (self.tetno_spoczynkowe - 40.0) / 40.0
        self.kondycja = max(0.0, min(1.0, wspolczynnik_kondycji))

    def aktualizuj_stan(self, poziom_wysilku: int, tryb_pauzy: bool = False) -> int:
        """Wylicza kolejny krok czasowy (takt 1 sekundy) stanu fizjologicznego."""
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

        # Ograniczenia bezpieczeństwa fizjologicznego
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

    def generuj_rekomendacje(self, hr: float, elapsed_s: int, tryb_pauzy: bool = False) -> Tuple[str, str]:
        if tryb_pauzy:
            return "TRYB REGENERACJI (PAUZA): Trwa proces schładzania organizmu (Cool-down). Tętno płynnie spada.", "Info"

        elapsed_min = elapsed_s / 60.0
        progress = (elapsed_min / self.target_duration) * 100

        if progress > 100:
            return "SESJA ZAKOŃCZONA. Utrzymaj powolne tempo do schłodzenia organizmu.", "Info"

        if self.bmi >= 28.0:
            msg_prefix = "OCHRONA STAWÓW (Wysokie BMI): "
        elif self.target_duration <= 20:
            msg_prefix = "SESJA KRÓTKA (HIIT): "
        else:
            msg_prefix = "SESJA WYTRZYMAŁOŚCIOWA: "

        # Obliczenie aktualnego obciążenia serca (% rezerwy tętna HRR)
        aktualne_hrr = (hr - self.tetno_spoczynkowe) / self.hrr if self.hrr > 0 else 0.0

        # Klasyfikacja aktualnych parametrów przez model scikit-learn
        biezace_dane = [[aktualne_hrr, self.bmi, self.target_duration]]
        decyzja_ai = self.model_ml.predict(biezace_dane)[0]

        if decyzja_ai == "ZWOLNIJ":
            return f"{msg_prefix}Zbyt wysokie tętno. Decyzja modelu ML: ZWOLNIJ.", "Error"
        elif decyzja_ai == "PRZYSPIESZ":
            return f"{msg_prefix}Tętno poniżej celu treningowego. Decyzja modelu ML: PRZYSPIESZ.", "Warning"
        else:
            return f"{msg_prefix}Parametry optymalne w wyznaczonej strefie wysiłkowej. Decyzja modelu ML: UTRZYMAJ TEMPO.", "Success"


# Konfiguracja głowna interface STEAMLIT
st.set_page_config(layout="wide", page_title="Dashboard Telemetrii Sportowej")
st.title("System Monitorowania Obciążeń Treningowych w Czasie Rzeczywistym")

# Inicjalizacja stanu sesji
if 'live' not in st.session_state: st.session_state.live = False
if 'pauza' not in st.session_state: st.session_state.pauza = False
if 'poziom_wysilku' not in st.session_state: st.session_state.poziom_wysilku = 0
if 'sesja_zakonczona' not in st.session_state: st.session_state.sesja_zakonczona = False
if 'hist_hr' not in st.session_state: st.session_state.hist_hr = []
if 'hist_t' not in st.session_state: st.session_state.hist_t = []

# Panel boczny
st.sidebar.header("Konfiguracja Zawodnika")
plec = st.sidebar.selectbox("Płeć", ["Mężczyzna", "Kobieta"])
wiek = st.sidebar.number_input("Wiek", min_value=15, max_value=90, value=25)
waga = st.sidebar.number_input("Masa (kg)", value=78.0, step=0.5)
wzrost = st.sidebar.number_input("Wzrost (cm)", value=180.0, step=0.5)
hr_rest = st.sidebar.number_input("Tętno spoczynkowe [BPM]", value=55)
plan_min = st.sidebar.slider("Planowany czas sesji (min)", 5, 120, 30)

st.sidebar.divider()
st.sidebar.header("Sterowanie Sesją")

if st.sidebar.button("URUCHOM SESJĘ", use_container_width=True):
    st.session_state.live = True
    st.session_state.pauza = False
    st.session_state.sesja_zakonczona = False
    st.session_state.poziom_wysilku = 1
    st.session_state.hist_hr = []
    st.session_state.hist_t = []
    st.session_state.sportowiec = WirtualnySportowiec(wiek, waga, wzrost, plec, hr_rest)
    s_obj = st.session_state.sportowiec
    st.session_state.asystent = AsystentAI(plan_min, s_obj.bmi, s_obj.hrr, s_obj.tetno_spoczynkowe)

col_p, col_s = st.sidebar.columns(2)

if col_p.button("PAUZA", use_container_width=True):
    if st.session_state.live and not st.session_state.pauza:
        st.session_state.pauza = True
    elif st.session_state.live and st.session_state.pauza:
        st.session_state.pauza = False

if col_s.button("STOP", use_container_width=True):
    st.session_state.live = False
    st.session_state.pauza = False
    st.session_state.sesja_zakonczona = True

if st.sidebar.button("Nowy trening (Reset)", use_container_width=True):
    st.session_state.live = False
    st.session_state.pauza = False
    st.session_state.sesja_zakonczona = False
    st.session_state.poziom_wysilku = 0
    st.session_state.hist_hr = []
    st.session_state.hist_t = []
    if 'sportowiec' in st.session_state:
        del st.session_state.sportowiec

# Instancjonowanie obiektów do podgladu live
if 'sportowiec' not in st.session_state:
    st.session_state.sportowiec = WirtualnySportowiec(wiek, waga, wzrost, plec, hr_rest)
if 'asystent' not in st.session_state:
    st.session_state.asystent = AsystentAI(plan_min, st.session_state.sportowiec.bmi, st.session_state.sportowiec.hrr, st.session_state.sportowiec.tetno_spoczynkowe)

s = st.session_state.sportowiec
ai = st.session_state.asystent

# Synchronizacja parametrów z paskiem bocznym
s.aktualizuj_parametry(wiek, waga, wzrost, plec, hr_rest)
ai.aktualizuj_parametry(plan_min, s.bmi, s.hrr, s.tetno_spoczynkowe)


# Panel główny
if st.session_state.live and not st.session_state.pauza:
    st.subheader("Panel Interwencji Trenerskiej")
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    if col_btn1.button("ZWOLNIJ", use_container_width=True):
        st.session_state.poziom_wysilku = max(0, st.session_state.poziom_wysilku - 1)
    if col_btn2.button("UTRZYMAJ TEMPO", use_container_width=True):
        pass
    if col_btn3.button("PRZYSPIESZ", use_container_width=True):
        st.session_state.poziom_wysilku = min(4, st.session_state.poziom_wysilku + 1)

    etykiety_wysilku = ["Spoczynek", "Lekki Trucht", "Bieg optymalny", "Mocny Bieg", "Maksymalny Sprint"]

    if s.niestabilnosc_tempa < 0.8:
        profil = "Znakomite (Tętno bardzo stabilne)"
    elif s.niestabilnosc_tempa < 1.3:
        profil = "Przeciętne (Naturalne wahania)"
    else:
        profil = "Niskie (Tętno rwane, wymaga częstych interwencji)"

    st.caption(f"Aktualnie narzucone tempo: **{etykiety_wysilku[st.session_state.poziom_wysilku]}** | Ukryty profil zawodnika: **{profil}**")

elif st.session_state.pauza:
    st.warning("Sesja wstrzymana (Tryb Aktywnego Schładzania). Kliknij ponownie 'PAUZA', aby wznowić obciążenie.")
elif not st.session_state.live and not st.session_state.sesja_zakonczona:
    st.info("ℹ️ **Tryb podglądu stref tętna:** Modyfikuj parametry w panelu bocznym, aby obserwować dynamikę zmian stref Karvonena oraz tętna maksymalnego. Kliknij **URUCHOM SESJĘ**, aby rozpocząć rejestrację telemetrii.")

# Symulacja kroku czasowego
if st.session_state.live:
    hr_now = s.aktualizuj_stan(st.session_state.poziom_wysilku, tryb_pauzy=st.session_state.pauza)
    st.session_state.hist_hr.append(hr_now)
    st.session_state.hist_t.append(s.czas_treningu_s)
else:
    hr_now = st.session_state.hist_hr[-1] if st.session_state.hist_hr else int(hr_rest)

avg_hr = int(sum(st.session_state.hist_hr) / len(st.session_state.hist_hr)) if st.session_state.hist_hr else int(hr_rest)
max_hr_achieved = max(st.session_state.hist_hr) if st.session_state.hist_hr else int(hr_rest)
trimp = int((s.czas_treningu_s / 60) * (avg_hr / s.max_tetno) * 1.5) if s.max_tetno > 0 else 0

st.divider()
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Czas sesji", f"{s.czas_treningu_s // 60}:{s.czas_treningu_s % 60:02d} / {plan_min}:00")
m2.metric("Tętno AVG", f"{avg_hr} BPM")
m3.metric("Tętno MAX (Limit)", f"{s.max_tetno} BPM")
m4.metric("Spalone Kalorie", f"{int(s.spalone_kcal)} kcal")
m5.metric("Obciążenie (TRIMP)", trimp)
m6.metric("BMI", f"{s.bmi:.1f}")

if st.session_state.live:
    msg, style = ai.generuj_rekomendacje(hr_now, s.czas_treningu_s, tryb_pauzy=st.session_state.pauza)
    if style == "Error":
        st.error(msg)
    elif style == "Warning":
        st.warning(msg)
    elif style == "Success":
        st.success(msg)
    else:
        st.info(msg)

# Wizualizacja PLOTLY
fig = go.Figure()
teksty_etykiet = [""] * len(st.session_state.hist_hr)
if teksty_etykiet:
    teksty_etykiet[-1] = f" {hr_now} BPM"

fig.add_trace(go.Scatter(
    x=st.session_state.hist_t,
    y=st.session_state.hist_hr,
    mode='lines+text',
    name='Tętno Live',
    line=dict(color='white', width=3),
    text=teksty_etykiet,
    textposition="middle right",
    textfont=dict(color="white", size=14, family="Arial Black"),
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
    (k_50, k_60, "Strefa 1 (50-60% HRR): Regeneracja", "#808080"),
    (k_60, k_70, "Strefa 2 (60-70% HRR): Spalanie tłuszczu", "#1f77b4"),
    (k_70, k_80, "Strefa 3 (70-80% HRR): Aerobowa", "#2ca02c"),
    (k_80, k_90, "Strefa 4 (80-90% HRR): Anaerobowa", "#ff7f0e"),
    (k_90, s.max_tetno, "Strefa 5 (90-100% HRR): Maksymalna / VO2 Max", "#d62728")
]

for low, high, label, color in zones:
    if high > low:
        fig.add_hrect(y0=low, y1=high, fillcolor=color, opacity=0.15, layer="below", line_width=0,
                      annotation_text=label)

linie_graniczne = [k_60, k_70, k_80, k_90]
for granica in linie_graniczne:
    fig.add_hline(
        y=granica,
        line_dash="dash",
        line_color="rgba(255, 255, 255, 0.3)",
        line_width=1.5,
        layer="below"
    )

fig.update_layout(
    height=480,
    margin=dict(l=70, r=90, t=30, b=60),
    xaxis=dict(
        title="Czas trwania",
        ticksuffix=" sek",
        showticklabels=True,
        tickfont=dict(color="#ffffff", size=12),
        title_font=dict(color="#aaaaaa", size=12),
        gridcolor="rgba(255, 255, 255, 0.15)"
    ),
    yaxis=dict(
        range=[40, max(210, s.max_tetno + 10)],
        title="Tętno",
        ticksuffix=" BPM",
        showticklabels=True,
        tickfont=dict(color="#ffffff", size=12),
        title_font=dict(color="#aaaaaa", size=12),
        gridcolor="rgba(255, 255, 255, 0.15)"
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    hoverlabel=dict(
        bgcolor="#1a1c23",
        font_size=14,
        font_family="Arial"
    )
)

st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# Raport HTML
if st.session_state.sesja_zakonczona:
    st.write("---")
    st.subheader("Eksport i Pobieranie Raportu Treningowego")

    with st.form(key="formularz_pobierania"):
        nazwa_treningu = st.text_input("Wprowadź nazwę pliku raportu:", value="Raport_Treningowy")
        zatwierdzono = st.form_submit_button("📁 PRZYGOTUJ RAPORT DO POBRANIA", use_container_width=True)

    if zatwierdzono:
        czysta_nazwa = nazwa_treningu.strip().replace(" ", "_")
        if not czysta_nazwa:
            czysta_nazwa = "Raport_Treningowy"

        plik_nazwa = f"{czysta_nazwa}.html"

        wykres_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

        raport_html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Raport Telemetryczny - {czysta_nazwa}</title>
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background-color: #0e1117; color: #ffffff; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }}
                .header h1 {{ margin: 0; color: #ff4b4b; font-size: 28px; }}
                .header p {{ color: #888; margin: 5px 0 0 0; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 15px; margin-bottom: 40px; }}
                .stat-card {{ background-color: #1a1c23; border: 1px solid #333; padding: 15px; border-radius: 8px; text-align: center; }}
                .stat-label {{ font-size: 11px; color: #888; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.5px; }}
                .stat-value {{ font-size: 24px; font-weight: bold; margin: 0; }}
                .chart-container {{ background-color: #1a1c23; border: 1px solid #333; border-radius: 8px; padding: 20px; }}
                .chart-title {{ font-size: 16px; margin-top: 0; margin-bottom: 15px; color: #ccc; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Raport z Telemetrii Wysiłkowej: {czysta_nazwa}</h1>
                    <p>Wygenerowano automatycznie przez System Monitorowania Obciążeń (Model Karvonena & Edwardsa)</p>
                </div>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Czas Sesji</div>
                        <div class="stat-value">{s.czas_treningu_s // 60}:{s.czas_treningu_s % 60:02d}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Tętno AVG</div>
                        <div class="stat-value">{avg_hr} BPM</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Tętno MAX</div>
                        <div class="stat-value">{max_hr_achieved} BPM</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Spalone Kalorie</div>
                        <div class="stat-value">{int(s.spalone_kcal)} kcal</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Obciążenie TRIMP</div>
                        <div class="stat-value">{trimp}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Wskaźnik BMI</div>
                        <div class="stat-value">{s.bmi:.1f}</div>
                    </div>
                </div>
                <div class="chart-container">
                    <div class="chart-title">Zapis telemetryczny krzywej tętna (Interaktywny)</div>
                    {wykres_html}
                </div>
            </div>
        </body>
        </html>
        """

        st.session_state.pobierz_dane = raport_html_content
        st.session_state.pobierz_nazwa = plik_nazwa

    if 'pobierz_dane' in st.session_state and st.session_state.pobierz_dane:
        st.success(f"Raport gotowy do pobrania: **{st.session_state.pobierz_nazwa}**")
        st.download_button(
            label="📥 KLIKNIJ, ABY POBRAĆ PLIK HTML",
            data=st.session_state.pobierz_dane,
            file_name=st.session_state.pobierz_nazwa,
            mime="text/html",
            use_container_width=True
        )

if st.session_state.live:
    time.sleep(1)
    st.rerun()
