import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import io
import time # [추가] 순차적 애니메이션을 그리기 위한 시간 라이브러리 추가

st.set_page_config(page_title="[생산성 예측 AI with Na]", page_icon="🌱", layout="centered")

@st.cache_data
def load_real_data():
    csv_data = """Lot,Day,pH,당농도(Bx),DCW(건조중량),Product
1,0,5.00,6.0,6.0,0.0
1,4,5.01,5.5,7.8,0.5
1,7,5.08,5.1,10.2,0.6
1,11,5.08,4.5,11.8,1.2
1,14,5.12,3.6,15.4,3.1
1,19,5.19,1.9,19.0,14.5
1,21,5.60,1.3,20.6,21.0
1,26,5.53,3.9,20.6,29.7
1,28,5.63,3.5,21.6,34.1
1,32,5.41,3.1,24.4,41.6
1,35,5.38,2.8,26.2,42.9
1,39,5.20,2.3,29.4,69.3
1,42,5.25,1.8,29.8,91.4
1,46,5.51,3.7,33.8,85.4
1,49,5.50,3.5,35.2,107.0
1,53,5.37,3.1,32.2,118.6
1,56,5.35,2.8,32.2,130.0
1,60,5.59,2.6,34.6,127.4
1,63,5.58,1.8,34.0,125.5
1,67,5.48,1.4,36.6,120.1
1,68,5.50,1.1,36.2,158.4
2,0,4.90,6.0,6.0,0.0
2,1,4.92,5.9,6.6,10.1
2,5,4.94,5.6,8.2,19.2
2,8,4.97,5.2,10.6,42.2
2,12,5.01,4.6,12.8,95.0
2,15,5.04,3.9,14.2,134.3
2,19,5.11,3.0,15.8,177.5
2,22,5.17,2.3,17.4,236.3
2,23,5.20,2.1,18.2,240.0
2,26,5.23,3.9,19.6,242.2
2,29,5.27,3.7,21.0,301.4
2,33,5.31,3.3,21.8,298.4
2,36,5.36,2.8,22.6,313.0
2,40,5.40,2.2,23.6,320.2
2,43,5.44,1.9,24.4,367.6
2,44,5.46,1.7,25.0,370.0
2,47,5.47,3.4,25.2,370.1
2,50,5.50,3.3,25.6,354.6
2,54,5.51,3.1,26.0,377.9
2,57,5.53,2.9,26.2,372.1
2,61,5.53,2.6,26.8,373.0
2,64,5.55,2.3,26.6,388.3
2,68,5.56,2.1,26.8,395.7
3,0,4.90,6.0,6.0,0.0
3,4,4.91,5.6,8.0,5.7
3,7,4.93,5.3,10.2,5.4
3,11,4.95,5.0,11.6,7.7
3,14,4.98,4.7,13.2,15.5
3,18,5.03,4.1,15.2,51.2
3,21,5.08,3.3,16.4,76.8
3,25,5.14,1.9,18.6,127.8
3,28,5.17,3.9,19.8,141.4
3,32,5.21,3.5,21.4,161.3
3,35,5.24,3.2,22.8,180.0
3,39,5.29,2.1,24.8,195.1
3,41,5.34,1.7,25.8,200.0
3,42,5.36,3.4,26.2,210.4
3,46,5.40,3.1,26.8,223.4
3,49,5.43,2.8,27.8,262.6
3,53,5.47,2.4,28.2,278.8"""
    df = pd.read_csv(io.StringIO(csv_data))
    return df

@st.cache_resource
def train_model(data):
    # [수정] Day 외에 엘리시터와 당 첨가 조건도 학습하도록 추가
    X = data[['Day', '엘리시터_처리일', '엘리시터_농도', '당_첨가일', '당_첨가량']] 
    y = data[['pH', '당농도(Bx)', 'DCW(건조중량)', 'Product']] 
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

@st.cache_resource
def train_inverse_sugar_model(data):
    X = data[['당농도(Bx)']]
    y = data[['Day']]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y.values.ravel())
    return model

@st.cache_resource
def train_inverse_dcw_model(data):
    X = data[['DCW(건조중량)']]
    y = data[['Day']]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y.values.ravel())
    return model

@st.cache_resource
def train_inverse_product_model(data):
    X = data[['Product']]
    y = data[['Day']]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y.values.ravel())
    return model

st.title("🌱 [생산성 예측 AI with Na]")
st.markdown("""
이 앱은 연구자님께서 직접 진행하신 **Lot 1, 2, 3의 실제 배양 데이터**를 학습한 인공지능 모델입니다.  
원하는 기능을 아래 탭에서 선택해 주세요.
""")

# --- 파일 업로드 기능 추가 ---
uploaded_file = st.file_uploader("📁 연구 데이터(CSV 또는 엑셀)를 업로드하세요", type=['csv', 'xlsx'])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        # CSV 파일의 경우 한글 깨짐을 방지하기 위해 인코딩 처리(필요시 encoding='cp949' 등 추가 가능)
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.success(f"**{uploaded_file.name}** 파일 업로드 완료! AI가 이 데이터를 기반으로 즉시 재학습했습니다.")
else:
    df = load_real_data()
    st.info("💡 준비하신 `2026-09-18T05-02_export.csv` 파일을 위 네모 칸 안에 끌어다 놓으세요! (현재는 기본 내장 데이터로 동작 중)")
# ---------------------------

# [추가] 기존 데이터에 엘리시터 및 당 첨가 열이 없다면 기본값(0)으로 채워줍니다.
# 연구자님이 엑셀에 이 4개의 열을 추가하시면 AI가 자동으로 인식해서 학습합니다!
for col in ['엘리시터_처리일', '엘리시터_농도', '당_첨가일', '당_첨가량']:
    if col not in df.columns:
        df[col] = 0.0

# 모델 4가지 모두 학습 진행
ai_model = train_model(df)
inv_sugar_model = train_inverse_sugar_model(df)
inv_dcw_model = train_inverse_dcw_model(df)
inv_product_model = train_inverse_product_model(df)

# 기존 3개 탭에서 4개 탭으로 확장
tab1, tab2, tab3, tab4 = st.tabs(["🗓️ 배양일수로 예측", "🎯 목표값 도달시기", "📈 실시간 궤적 시뮬레이션", "📊 원본 데이터"])

# --- 탭 1: 기존 기능 (배양일수 입력) ---
with tab1:
    st.subheader("🧪 특정 일차(Day)의 상태를 예측합니다")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        input_days = st.number_input("예측할 배양일수(Day)", min_value=0, max_value=100, value=30, step=1)
        # [추가] 엘리시터 및 당 첨가 조건 입력칸
        el_day = st.number_input("엘리시터 처리일(Day)", min_value=0, max_value=100, value=0)
        el_conc = st.number_input("엘리시터 농도(mg/L)", min_value=0.0, value=0.0, step=0.1)
        sugar_day = st.number_input("당 첨가일(Day)", min_value=0, max_value=100, value=0)
        sugar_amt = st.number_input("당 첨가량(g/L)", min_value=0.0, value=0.0, step=1.0)

    with col2:
        st.write("") 
        st.write("")
        if st.button("🚀 결과 예측하기", key="btn_forward", use_container_width=True):
            # [수정] 새로운 조건을 포함하여 예측
            user_input = pd.DataFrame({
                'Day': [input_days],
                '엘리시터_처리일': [el_day],
                '엘리시터_농도': [el_conc],
                '당_첨가일': [sugar_day],
                '당_첨가량': [sugar_amt]
            })
            prediction = ai_model.predict(user_input)
            
            pred_ph = prediction[0][0]
            pred_sugar = prediction[0][1]
            pred_dcw = prediction[0][2]
            pred_product = prediction[0][3]

            st.success(f"**Day {input_days}** 의 생산성 예측 결과입니다!")
            
            res_col1, res_col2 = st.columns(2)
            res_col3, res_col4 = st.columns(2)
            with res_col1: st.metric(label="예상 pH", value=f"{pred_ph:.2f}")
            with res_col2: st.metric(label="예상 당농도(Bx)", value=f"{pred_sugar:.2f}")
            with res_col3: st.metric(label="예상 DCW(건조중량)", value=f"{pred_dcw:.1f} g/L")
            with res_col4: st.metric(label="예상 Product", value=f"{pred_product:.1f}")

# --- 탭 2: 신규 기능 (역추산) ---
with tab2:
    st.subheader("🎯 원하는 상태가 되려면 며칠이 걸릴지 예측합니다")
    
    # 당농도로 예측할지 DCW로 예측할지 Product로 예측할지 선택
    target_type = st.radio("어떤 값을 목표로 설정하시겠습니까?", ["당농도(Bx) 도달 시기", "DCW(건조중량) 도달 시기", "Product 도달 시기"], horizontal=True)
    
    st.write("---")
    
    col3, col4 = st.columns([1, 2])
    with col3:
        if "당농도" in target_type:
            target_val = st.number_input("목표 당농도(Bx)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
            btn_label = "🚀 당농도 도달 시기 예측"
        elif "DCW" in target_type:
            target_val = st.number_input("목표 DCW(g/L)", min_value=0.0, max_value=50.0, value=25.0, step=1.0)
            btn_label = "🚀 DCW 도달 시기 예측"
        else:
            target_val = st.number_input("목표 Product", min_value=0.0, max_value=500.0, value=150.0, step=1.0)
            btn_label = "🚀 Product 도달 시기 예측"
            
    with col4:
        st.write("") 
        st.write("")
        if st.button(btn_label, key="btn_inverse", use_container_width=True):
            if "당농도" in target_type:
                # 당농도로 일수 예측
                user_input_inv = pd.DataFrame({'당농도(Bx)': [target_val]})
                pred_day = inv_sugar_model.predict(user_input_inv)[0]
                target_str = f"당농도 {target_val} Bx"
            elif "DCW" in target_type:
                # DCW로 일수 예측
                user_input_inv = pd.DataFrame({'DCW(건조중량)': [target_val]})
                pred_day = inv_dcw_model.predict(user_input_inv)[0]
                target_str = f"DCW {target_val} g/L"
            else:
                # Product로 일수 예측
                user_input_inv = pd.DataFrame({'Product': [target_val]})
                pred_day = inv_product_model.predict(user_input_inv)[0]
                target_str = f"Product {target_val}"
            
            st.success(f"목표한 **{target_str}**에 도달하기 위해 예상되는 배양일수는 약 **{pred_day:.1f} 일** 입니다!")
            
            # [수정] 예측된 날짜에 다른 수치들은 어떨지도 서비스로 보여줍니다. (나머지 4개 변수 기본값 0으로 추가)
            day_input = pd.DataFrame({
                'Day': [pred_day],
                '엘리시터_처리일': [0],
                '엘리시터_농도': [0.0],
                '당_첨가일': [0],
                '당_첨가량': [0.0]
            })
            prediction_for_that_day = ai_model.predict(day_input)
            
            st.info(f"💡 참고: 예상된 {pred_day:.1f}일 차의 다른 지표들은 **pH {prediction_for_that_day[0][0]:.2f}, Product {prediction_for_that_day[0][3]:.1f}** 수준일 것으로 추정됩니다.")

# --- 탭 3: 실시간 데이터 궤적 시뮬레이션 (애니메이션 그래프) ---
with tab3:
    st.subheader("📈 현재 배양 상태를 입력하면 70일까지의 궤적을 예측합니다")
    st.write("중간에 측정한 특정일의 데이터를 입력하시면, AI가 측정값을 기준으로 남은 기간의 성장 곡선을 순차적으로 시뮬레이션합니다.")
    
    # [수정] 입력 폼을 두 줄로 나누어 엘리시터와 당 첨가 계획도 입력받습니다.
    st.markdown("**1. 현재 측정값 입력**")
    col_d, col_w, col_p, col_pr, col_b = st.columns(5)
    with col_d: in_day = st.number_input("현재 일수(Day)", min_value=0, max_value=69, value=15, step=1)
    with col_w: in_dcw = st.number_input("현재 DCW", value=14.2)
    with col_p: in_ph = st.number_input("현재 pH", value=5.04)
    with col_pr: in_prod = st.number_input("현재 Product", value=134.3)
    with col_b: in_bx = st.number_input("현재 당농도", value=3.9)

    st.markdown("**2. 향후 공정 계획 입력 (엘리시터 / 당 첨가)**")
    col_el1, col_el2, col_sg1, col_sg2 = st.columns(4)
    with col_el1: sim_el_day = st.number_input("예정 엘리시터 처리일", value=20, step=1)
    with col_el2: sim_el_conc = st.number_input("예정 엘리시터 농도", value=1.0, step=0.1)
    with col_sg1: sim_sg_day = st.number_input("예정 당 첨가일", value=30, step=1)
    with col_sg2: sim_sg_amt = st.number_input("예정 당 첨가량", value=10.0, step=1.0)
    
    if st.button("▶️ 70일까지 궤적 시뮬레이션 시작", use_container_width=True):
        st.success(f"Day {in_day}부터 Day 70까지의 예측 궤적을 렌더링합니다...")
        
        # 1. 인공지능의 기본 예측값과 현재 측정값의 차이 계산
        # [수정] 예측 기준에 예정된 공정 조건 포함
        base_input = pd.DataFrame({
            'Day': [in_day],
            '엘리시터_처리일': [sim_el_day],
            '엘리시터_농도': [sim_el_conc],
            '당_첨가일': [sim_sg_day],
            '당_첨가량': [sim_sg_amt]
        })
        base_pred = ai_model.predict(base_input)[0]
        diff_ph = in_ph - base_pred[0]
        diff_bx = in_bx - base_pred[1]
        diff_dcw = in_dcw - base_pred[2]
        diff_prod = in_prod - base_pred[3]
        
        g_col1, g_col2 = st.columns(2)
        g_col3, g_col4 = st.columns(2)
        
        with g_col1: 
            st.markdown("**🌱 DCW (건조중량) 예측 궤적**")
            chart_dcw = st.empty() # 빈 공간 생성
        with g_col2: 
            st.markdown("**🧪 Product 예측 궤적**")
            chart_prod = st.empty() # 빈 공간 생성
        with g_col3: 
            st.markdown("**💧 pH 예측 궤적**")
            chart_ph = st.empty() # 빈 공간 생성
        with g_col4: 
            st.markdown("**🍬 당농도(Bx) 예측 궤적**")
            chart_bx = st.empty() # 빈 공간 생성
            
        # 데이터를 누적할 빈 상자(데이터프레임) 준비 및 첫날 데이터 넣기
        df_dcw = pd.DataFrame({'DCW': [in_dcw]}, index=[in_day])
        df_prod = pd.DataFrame({'Product': [in_prod]}, index=[in_day])
        df_ph = pd.DataFrame({'pH': [in_ph]}, index=[in_day])
        df_bx = pd.DataFrame({'당농도(Bx)': [in_bx]}, index=[in_day])

        # 첫 화면에 점 하나 그리기
        chart_dcw.line_chart(df_dcw, height=200)
        chart_prod.line_chart(df_prod, height=200)
        chart_ph.line_chart(df_ph, height=200)
        chart_bx.line_chart(df_bx, height=200)
        
        for d in range(in_day + 1, 71):
            # [수정] 매일매일의 궤적을 그릴 때 공정 계획 반영
            sim_input = pd.DataFrame({
                'Day': [d],
                '엘리시터_처리일': [sim_el_day],
                '엘리시터_농도': [sim_el_conc],
                '당_첨가일': [sim_sg_day],
                '당_첨가량': [sim_sg_amt]
            })
            p = ai_model.predict(sim_input)[0]
            
            # 측정값과 모델예측값의 차이를 반영, 음수가 나오지 않게 최소값은 0으로 제한
            adj_ph = max(0.0, p[0] + diff_ph)
            adj_bx = max(0.0, p[1] + diff_bx)
            adj_dcw = max(0.0, p[2] + diff_dcw)
            adj_prod = max(0.0, p[3] + diff_prod)
            
            # 다음 날짜의 데이터를 기존 데이터에 누적해서 합치기
            df_dcw = pd.concat([df_dcw, pd.DataFrame({'DCW': [adj_dcw]}, index=[d])])
            df_prod = pd.concat([df_prod, pd.DataFrame({'Product': [adj_prod]}, index=[d])])
            df_ph = pd.concat([df_ph, pd.DataFrame({'pH': [adj_ph]}, index=[d])])
            df_bx = pd.concat([df_bx, pd.DataFrame({'당농도(Bx)': [adj_bx]}, index=[d])])
            
            # 빈 공간에 새로운 차트를 덮어쓰기 (애니메이션 효과)
            chart_dcw.line_chart(df_dcw, height=200)
            chart_prod.line_chart(df_prod, height=200)
            chart_ph.line_chart(df_ph, height=200)
            chart_bx.line_chart(df_bx, height=200)
            
            # 0.05초 기다렸다가 다음 날짜 그리기 (속도 조절)
            time.sleep(0.05)
            
        st.info("🎯 시뮬레이션이 완료되었습니다! 그래프에 마우스를 올리시면 일차(Day)별 정확한 수치를 확인할 수 있습니다.")

# --- 탭 4: 데이터 확인 ---
with tab4:
    st.write("AI가 학습한 총 61개의 실험 기록입니다.")
    st.dataframe(df, use_container_width=True)


