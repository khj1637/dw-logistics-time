# 1. 라이브러리 및 설정
import streamlit as st
import numpy as np
import pandas as pd
import re
import base64
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image
from io import BytesIO
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import RobustScaler

st.markdown(
    """
    <style>
    /* 상단 'Made with Streamlit' 로고 숨김 */
    header {visibility: hidden;}
    
    /* 하단 footer 숨김 (버튼 포함) */
    footer {visibility: hidden;}
    
    /* 특정 클래스명 요소 숨김 */
    ._profileContainer_gzau3_53 {display: none !important;}

    /* 특정 클래스명 요소 숨김 */
    ._container_gzau3_1 _viewerBadge_nim44_23 {display: none !important;}
    </style>
    """,
    unsafe_allow_html=True
)


# 폰트 설정
font_path = "./NanumGothic.ttf"  # 또는 "./fonts/NanumGothic.ttf"
fontprop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = fontprop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# 2. 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv("data_logistics.csv")
    df = df.rename(columns={"현장명": "프로젝트명"})
    return df

df_data = load_data()

st.markdown(
    """
    <div style="text-align: center; margin-bottom: 5px;">
        <img src="https://raw.githubusercontent.com/khj1637/dw-workday-ai/main/img/logo.png"
             alt="DongwonCI"
             width="180"
             style="display: block; margin: auto; padding-bottom: 5px;">
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h1 style='text-align: center;'>물류센터 공사기간 예측기</h1>
    <div style='height: 20px;'></div>  <!-- 공백 한 줄 -->
    <p style='text-align: left; font-size: 0.85rem; color: #555;'>
        버전: v1.0.0<br>
        최종 업데이트: 2025년 6월 17일<br>
        개발자 : 동원건설산업 기술팀 김혁진
    </p>
    """,
    unsafe_allow_html=True
)

# 4. 보조 함수들

def detect_outlier_floors(floor_count):
    if floor_count >= 50:
        st.warning("⚠️ 입력하신 층수가 비정상적으로 높습니다. 다시 확인해주세요.")

def get_star_score(r2, std=None):
    score = r2 * 5
    if std is not None and std > 5:
        score -= 1
    score = max(0, min(5, score))
    return "★" * int(round(score)) + "☆" * (5 - int(round(score)))

@st.cache_data
def knn_impute(df_input, n_neighbors=3):
    numeric_cols = df_input.select_dtypes(include=["float", "int"]).columns
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_filled = df_input.copy()
    df_filled[numeric_cols] = imputer.fit_transform(df_input[numeric_cols])
    return df_filled

# 5. 유사도 기반 예측 함수 (RandomForest 기반 가중치 적용)
def find_similar_projects_with_categorical_filter(user_row, df, top_k=5):
    numeric_cols = ['대지면적', '건축면적', '연면적', '지하층수', '지상층수', '하역가능층수', '최고높이', '전체토공량']
    structure_cols = [col for col in df.columns if col.startswith('구조형식_')]
    coldtype_cols = [col for col in df.columns if col.startswith('창고유형_')]

    input_values = user_row.iloc[0]
    available_numeric = [col for col in numeric_cols if col in input_values.index and pd.notnull(input_values[col])]
    if not available_numeric:
        raise ValueError("입력값에 유효한 수치형 항목이 없습니다.")

    selected_coldtypes = [col for col in coldtype_cols if col in user_row.columns and input_values[col] == 1]
    coldtype_mask = df[coldtype_cols].eq(0)
    for col in selected_coldtypes:
        coldtype_mask[col] = df[col].eq(1)
    coldtype_match = coldtype_mask.all(axis=1)
    df_matched = df[coldtype_match].dropna(subset=available_numeric).copy()

    if df_matched.empty:
        raise ValueError("입력한 창고유형과 정확히 일치하는 프로젝트를 찾을 수 없습니다.")

    feature_cols = available_numeric + structure_cols
    df_model = df.dropna(subset=["전체공사기간"])
    X_model = df_model[feature_cols]
    y_model = df_model["전체공사기간"]
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_model, y_model)
    feature_importance = rf_model.feature_importances_
    weights = dict(zip(feature_cols, feature_importance))

    scaler = RobustScaler().fit(df_matched[feature_cols])
    df_scaled = scaler.transform(df_matched[feature_cols])
    user_scaled = scaler.transform(user_row[feature_cols])[0]

    def weighted_euclidean(a, b, w_dict, cols):
        return np.sqrt(np.sum([(a[i] - b[i]) ** 2 * w_dict[cols[i]] for i in range(len(cols))]))

    distances = np.array([weighted_euclidean(row, user_scaled, weights, feature_cols) for row in df_scaled])
    similarities = 1 / (distances + 1e-8)
    df_matched["유사도점수"] = similarities * 100

    top_similar = df_matched.sort_values("유사도점수", ascending=False).head(top_k)
    if "전체공사기간" not in top_similar.columns or top_similar["전체공사기간"].isnull().all():
        predicted_months = np.nan
    else:
        weighted_sum = (top_similar["전체공사기간"] * top_similar["유사도점수"]).sum()
        sum_weights = top_similar["유사도점수"].sum()
        predicted_months = weighted_sum / sum_weights

    return top_similar, round(predicted_months, 1) if not np.isnan(predicted_months) else None

# 6. 사용자 입력값 파싱 함수 추가
def parse_float(text):
    try:
        return float(text.replace(",", "").strip()) if text else np.nan
    except:
        return np.nan

def parse_int(text):
    try:
        return int(text.strip()) if text else np.nan
    except:
        return np.nan

# 7. 사용자 입력 UI
with st.expander("필수 입력값", expanded=True):
    
    # 👉 프로젝트명 입력 필드 추가
    

    col1, col2, col3 = st.columns(3)
    with col1:
        project_name = st.text_input("프로젝트명")
        floors_below_str = st.text_input("지하층수", placeholder="예: 1")
        
    with col2:
        floors_above_str = st.text_input("지상층수", placeholder="예: 4")
        land_area_str = st.text_input("대지면적 (㎡)", placeholder="예: 45085")
              
    with col3:
        building_area_str = st.text_input("건축면적 (㎡)", placeholder="예: 26166")
        total_area_str = st.text_input("연면적 (㎡)", placeholder="예: 70841")

    # 수치형 변환
    land_area = parse_float(land_area_str)
    building_area = parse_float(building_area_str)
    total_area = parse_float(total_area_str)
    floors_above = parse_int(floors_above_str)
    floors_below = parse_int(floors_below_str)

    detect_outlier_floors(floors_above)

    st.divider()
    st.markdown("#### 창고유형 (복수 선택 가능)")
    usage_options = ['상온', '저온', '냉동']
    selected_usages = []
    for i, u in enumerate(st.columns(len(usage_options))):
        if u.checkbox(usage_options[i], key=f"usage_{usage_options[i]}"):
            selected_usages.append(usage_options[i])

    st.markdown("#### 구조형식 (복수 선택 가능)")
    structure_options = ['PC', 'RC', 'SRC', 'PEB']
    selected_structures = []
    for i, s in enumerate(st.columns(len(structure_options))):
        if s.checkbox(structure_options[i], key=f"struct_{structure_options[i]}"):
            selected_structures.append(structure_options[i])

with st.expander("선택 입력값", expanded=True):
    col4, col5, col6 = st.columns(3)
    with col4:
        excavation_volume_str = st.text_input("전체 토공량 (㎥)", placeholder="예: 120000")
    with col5:
        dockable_floors_str = st.text_input("하역 가능층수", placeholder="예: 2")
    with col6:
        height_str = st.text_input("최고높이 (m)", placeholder="예: 47.5")

    excavation_volume = parse_float(excavation_volume_str)
    dockable_floors = parse_int(dockable_floors_str)
    height = parse_float(height_str)

# 8. 예측 실행 및 결과 출력

if st.button("예측 시작", use_container_width=True):
    # 필수 입력값 누락 체크
    required_fields = {
        "대지면적": land_area,
        "건축면적": building_area,
        "연면적": total_area,
        "지상층수": floors_above,
        "지하층수": floors_below,
        "창고유형": selected_usages,
        "구조형식": selected_structures
    }
    missing_fields = [k for k, v in required_fields.items() if (isinstance(v, list) and not v) or (not isinstance(v, list) and pd.isna(v))]
    if missing_fields:
        st.warning(f"⚠️ 필수 입력값이 누락되었습니다: {', '.join(missing_fields)} 항목을 입력해 주세요.")
        st.stop()

    # 논리적 유효성 검사
    logic_errors = []
    if building_area > land_area:
        logic_errors.append("건축면적이 대지면적보다 클 수 없습니다.")
    if total_area < building_area:
        logic_errors.append("연면적이 건축면적보다 작을 수 없습니다.")
    if (floors_above + floors_below) == 0 and total_area > 0:
        logic_errors.append("층수가 없는데 연면적이 존재할 수 없습니다.")
    if total_area > 0 and building_area > 0 and floors_above > 0:
        estimated_total = building_area * (floors_above + floors_below)
        if total_area > estimated_total * 1.5:
            logic_errors.append("연면적이 건축면적×층수보다 지나치게 큽니다. 입력값을 확인해 주세요.")
    if logic_errors:
        st.markdown("### ⚠️ 입력값 오류 감지")
        for err in logic_errors:
            st.error(f"🚫 {err}")
        st.stop()

    # 사용자 입력값 정리
    user_input = {
        "대지면적": land_area,
        "건축면적": building_area,
        "연면적": total_area,
        "지상층수": floors_above,
        "지하층수": floors_below,
        "하역가능층수": dockable_floors if dockable_floors and dockable_floors > 0 else np.nan,
        "최고높이": height if height and height > 0 else np.nan,
        "전체토공량": excavation_volume if excavation_volume and excavation_volume > 0 else np.nan,
    }
    input_df = pd.DataFrame([user_input])
    for s in structure_options:
        input_df[f"구조형식_{s}"] = 1 if s in selected_structures else 0
    for u in usage_options:
        input_df[f"창고유형_{u}"] = 1 if u in selected_usages else 0

    # 학습 데이터 준비
    user_corrected = input_df.copy()
    df_model = df_data.dropna(subset=["전체공사기간"])
    feature_cols = user_corrected.columns.tolist()
    X_all = df_model[feature_cols].copy()
    y_all = df_model["전체공사기간"]
    X_all = X_all.loc[:, X_all.isna().mean() <= 0.3]
    user_corrected = user_corrected[X_all.columns]

    # KNN 보간
    X_all_imputed = knn_impute(X_all)

    # 선형회귀
    model1 = LinearRegression().fit(X_all_imputed, y_all)
    pred1 = model1.predict(user_corrected)[0]
    r2_1 = model1.score(X_all_imputed, y_all)

    # 랜덤포레스트
    model2 = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_all_imputed, y_all)
    pred2 = model2.predict(user_corrected)[0]
    r2_2 = model2.score(X_all_imputed, y_all)

    # 유사 프로젝트 기반
    similar_df, pred3 = find_similar_projects_with_categorical_filter(user_corrected, df_data)
    sim_std = similar_df["전체공사기간"].std()
    mean_similarity = similar_df["유사도점수"].mean()

    # 앙상블
    pred_ensemble = (pred1 * r2_1 + pred2 * r2_2 + pred3 * 1.0) / (r2_1 + r2_2 + 1.0)

    # 신뢰도 검사
    warnings = []
    if sim_std > 6:
        warnings.append("유사 프로젝트 간 공사기간 편차가 커서 신뢰도가 낮습니다.")
    if mean_similarity < 25:
        warnings.append("입력 조건과 유사한 프로젝트가 적어 예측 정확도가 낮을 수 있습니다.")
    if r2_1 < 0.3 and r2_2 < 0.3:
        warnings.append("머신러닝 모델의 예측 신뢰도가 낮습니다.")
    if warnings:
        st.markdown("### ⚠️ 예측 결과 출력이 제한되었습니다.")
        for w in warnings:
            st.error(f"🚫 {w}")
        st.info("입력값을 수정하거나 보완한 후 다시 예측을 실행해 주세요.")
        st.stop()

    
    if project_name:
            # ⭐ 신뢰도 점수 계산 및 별점 변환
        trust_score = (r2_1 + r2_2 + 1.0) / 3
        star_rating = get_star_score(trust_score)
        
        st.markdown(f"""
            <div style="
                background-color: #f4f8fc;
                border: 1px solid #d0e3f1;
                border-radius: 12px;
                padding: 25px;
                margin-top: 20px;
                margin-bottom: 20px;
                box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.05);
            ">
                <div style="text-align: center; font-size: 1.6rem; font-weight: 700; color: #004080;">
                    📌 {project_name} 산출 결과
                </div>
                <div style="text-align: center; font-size: 2.4rem; font-weight: bold; color: #00264d; margin-top: 12px;">
                    {round(pred_ensemble, 1)} 개월
                </div>
                <div style="text-align: center; font-size: 0.95rem; color: #555; margin-top: 10px;">
                    최종 예측 결과는 <strong style="color:#004080;">머신러닝 (선형회귀, 랜덤포레스트)</strong>과
                    <strong style="color:#004080;">유사 프로젝트 기반</strong> 결과를 종합하여 계산되었습니다.
                </div>
                <div style="text-align: center; font-size: 0.9rem; color: #444; margin-top: 8px;">
                    해당 예측 결과의 <strong style="color:#004080;">신뢰도</strong>는 <span style="font-weight: bold;">{star_rating}</span> 입니다.
                </div>
            </div>
        """, unsafe_allow_html=True)

    # 예측 결과 표
    st.subheader("■ 예측 결과 요약")
    final_table = pd.DataFrame({
        "예측 방식": ["선형회귀", "랜덤포레스트", "유사 프로젝트 기반"],
        "예측값 (개월)": [round(pred1, 1), round(pred2, 1), round(pred3, 1)],
        "신뢰도": [
            get_star_score(r2_1),
            get_star_score(r2_2),
            get_star_score(1.0, sim_std)
        ]
    })
    st.dataframe(final_table)

    # 📊 예측 결과 비교표
    if not similar_df.empty:
        st.subheader("■ 예측 결과 그래프")

        fig, ax = plt.subplots(figsize=(10, 5))

        # 히스토그램: 1개월 단위, 부드러운 스타일
        min_month = int(similar_df["전체공사기간"].min()) - 1
        max_month = int(similar_df["전체공사기간"].max()) + 1
        bins = np.arange(min_month, max_month + 1)

        ax.hist(
            similar_df["전체공사기간"],
            bins=bins,
            alpha=0.6,
            color='#cccccc',  # 연한 회색
            edgecolor='#888888',  # 어두운 테두리
            linewidth=1,
            label='유사 프로젝트 분포',
            align='mid'
        )

        # 예측값 라인
        ax.axvline(pred1, color='blue', linestyle='--', linewidth=1.2, label='선형회귀 예측')
        ax.axvline(pred2, color='green', linestyle='--', linewidth=1.2, label='랜덤포레스트 예측')
        ax.axvline(pred3, color='orange', linestyle='--', linewidth=1.2, label='유사 기반 예측')
        ax.axvline(pred_ensemble, color='red', linestyle='-', linewidth=2.5, label='앙상블 예측')

        # 스타일 향상
        ax.grid(axis='y', linestyle=':', linewidth=0.7, alpha=0.5)
        ax.set_facecolor('#f9f9f9')  # 밝은 회색 배경

        # 폰트 및 레이아웃
        ax.set_xticks(np.arange(min_month, max_month + 1, 1))
        ax.set_xlabel("공사기간", fontproperties=fontprop)
        ax.set_ylabel("유사 프로젝트 수", fontproperties=fontprop)
        ax.set_title("공사기간 예측 결과 및 유사 프로젝트 분포", fontproperties=fontprop)

        # 범례 하단 가로 정렬
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=5, prop=fontprop, frameon=False)

        st.pyplot(fig)

    
    # 📉 신뢰도 기반 경고 메시지 출력
    warnings = []

    # 1. 유사 프로젝트 간 공사기간 표준편차가 너무 크면 → 신뢰 낮음
    if sim_std > 6:
        warnings.append("유사 프로젝트 간 공사기간 편차가 커서 신뢰도가 낮습니다.")

    # 2. 유사도 평균이 낮으면 → 유사 사례가 거의 없음
    mean_similarity = similar_df["유사도점수"].mean()
    if mean_similarity < 25:
        warnings.append("입력 조건과 유사한 프로젝트가 적어 예측 정확도가 낮을 수 있습니다.")

    # 3. 머신러닝 모델의 R² 점수가 둘 다 낮으면 → 일반화 성능 낮음
    if r2_1 < 0.3 and r2_2 < 0.3:
        warnings.append("머신러닝 모델의 예측 신뢰도가 낮습니다. 입력값을 다시 확인해 주세요.")

    # 실제 경고 메시지 출력
    if warnings:
        st.markdown("### ⚠️ 예측 결과 주의사항")
        for w in warnings:
            st.warning(w)

    # 유사 프로젝트 리스트
    st.subheader("■ 참조된 유사 프로젝트")
    if not similar_df.empty:
        display_df = similar_df.copy()

        def decode_types(row, prefix):
            cols = [col for col in row.index if col.startswith(prefix)]
            return " + ".join(col.replace(prefix, '') for col in cols if row[col] == 1)

        display_df["구조형식"] = display_df.apply(lambda row: decode_types(row, "구조형식_"), axis=1)
        display_df["창고유형"] = display_df.apply(lambda row: decode_types(row, "창고유형_"), axis=1)
        display_df["층수"] = display_df.apply(
            lambda row: f"B{int(row['지하층수']) if not pd.isna(row['지하층수']) else 0} / "
                        f"{int(row['지상층수']) if not pd.isna(row['지상층수']) else 0}F",
            axis=1
        )

        display_df = display_df[[
            "프로젝트명", "연면적", "층수", "구조형식", "창고유형", "전체공사기간"
        ]].rename(columns={
            "연면적": "연면적 (㎡)",
            "전체공사기간": "공사기간 (개월)"
        })

        st.dataframe(display_df)
        st.markdown("📌 **참조된 유사 프로젝트**는 입력 조건과 범주형 항목이 일치한 실제 사례들입니다.")
    else:
        st.warning("⚠️ 유사 프로젝트를 찾을 수 없습니다. 입력값을 다시 확인해 주세요.")


