import streamlit as st
import pandas as pd
import io

# --------------------------------------------------------------------------
# 1. 기본 설정 및 스타일
# --------------------------------------------------------------------------
st.set_page_config(page_title="2026학년도 반편성 프로그램", layout="wide")

st.markdown("""
    <style>
    .highlight { color: red; font-weight: bold; }
    .stAlert { padding: 0.5rem; }
    .stDataFrame { border: 1px solid #ddd; }
    </style>
""", unsafe_allow_html=True)

st.title("🏫 2026학년도 초등학교 반편성 시스템")
st.markdown("""
**반편성 원칙:**
1. **학반별 순환 배정:** 1반(가→나→다), 2반(나→다→가), 3반(다→가→나) 로테이션 적용
2. **S자형 성적 안배:** 성적 편차 최소화를 위해 S자(ㄹ자) 패턴 적용
3. **생활지도 집중 분산:** 반별 생활지도 학생 수가 균등해질 때까지(차이 1명 이내) 최적 교환
""")

# --------------------------------------------------------------------------
# 2. 데이터 처리 및 알고리즘 함수
# --------------------------------------------------------------------------
def preprocess_data(df):
    """데이터 정제"""
    col_map = {
        '성명': '이름',
        '합': '총점',
        '학반': '2025반',
        '번호': '2025번호',
        '생활지도 곤란': '생활지도'
    }
    df = df.rename(columns=col_map)
    
    required = ['이름', '성별', '총점', '2025반', '2025번호']
    if not all(col in df.columns for col in required):
        return None, f"필수 컬럼이 누락되었습니다. (필요: {required}, 현재: {list(df.columns)})"

    df = df.dropna(subset=['이름'])
    
    # 점수 처리
    df['총점'] = pd.to_numeric(df['총점'], errors='coerce')
    avg_score = df['총점'].mean()
    if pd.isna(avg_score): avg_score = 0 
    df['총점'] = df['총점'].fillna(avg_score).round().astype(int)
    
    # 반, 번호 처리
    for col in ['2025반', '2025번호']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # 생활지도 표시
    if '생활지도' in df.columns:
        df['생활지도_표시'] = df['생활지도'].astype(str).apply(
            lambda x: True if x.strip() not in ['nan', '', '0', '0.0', 'None'] else False
        )
    else:
        df['생활지도_표시'] = False
        
    return df, None

def allocate_class_logic(df):
    """학반별 순환 S자 배정"""
    results = []
    
    # 2025반과 성별로 그룹핑
    for (old_class, gender), sub_df in df.groupby(['2025반', '성별']):
        sub_df = sub_df.sort_values(by=['총점', '이름'], ascending=[False, True]).copy()
        
        # 순환 배정 로직
        if old_class == 1: targets = ['가', '나', '다']
        elif old_class == 2: targets = ['나', '다', '가']
        elif old_class == 3: targets = ['다', '가', '나']
        else: targets = ['가', '나', '다']
            
        new_classes = []
        for i in range(len(sub_df)):
            cycle = i % 6
            if cycle == 0: idx = 0
            elif cycle == 1: idx = 1
            elif cycle == 2: idx = 2
            elif cycle == 3: idx = 2
            elif cycle == 4: idx = 1
            else: idx = 0
            new_classes.append(targets[idx])
            
        sub_df['신학년반'] = new_classes
        results.append(sub_df)
        
    if not results: return df
    return pd.concat(results, ignore_index=True)

def distribute_special_students_global(df):
    """
    [생활지도 학생 강력 분산]
    모든 반의 생활지도 학생 수 차이가 1명 이하가 될 때까지 반복 교환
    """
    max_iter = 300 # 충분한 반복 횟수 보장
    
    for i in range(max_iter):
        # 1. 현재 상태 파악
        counts = df[df['생활지도_표시'] == True]['신학년반'].value_counts()
        for cls in ['가', '나', '다']:
            if cls not in counts: counts[cls] = 0
            
        max_val = counts.max()
        min_val = counts.min()
        
        # 2. 종료 조건: 차이가 1명 이하면 최적 상태 (예: 6,6,5)
        if max_val - min_val <= 1:
            break
            
        # 3. 과밀 학급(src)과 부족 학급(dst) 식별
        src_class = counts.idxmax()
        dst_class = counts.idxmin()
        
        # 4. 교환 가능한 최적의 쌍 찾기 (전수 조사)
        src_candidates = df[
            (df['신학년반'] == src_class) & 
            (df['생활지도_표시'] == True)
        ]
        
        best_swap_pair = None
        min_score_diff = float('inf')
        
        # 모든 후보를 검사
        for src_idx, src_student in src_candidates.iterrows():
            s_gender = src_student['성별']
            s_score = src_student['총점']
            
            # 부족한 반에서 '성별이 같고' '생활지도가 아닌' 학생 찾기
            dst_candidates = df[
                (df['신학년반'] == dst_class) & 
                (df['생활지도_표시'] == False) & 
                (df['성별'] == s_gender)
            ]
            
            if dst_candidates.empty:
                continue
            
            # 점수 차이 계산
            current_candidates = dst_candidates.copy()
            current_candidates['diff'] = abs(current_candidates['총점'] - s_score)
            
            # 가장 점수가 비슷한 학생
            best_match = current_candidates.sort_values('diff').iloc[0]
            current_diff = best_match['diff']
            
            # 역대급으로 점수 차가 적으면 후보 등록
            if current_diff < min_score_diff:
                min_score_diff = current_diff
                best_swap_pair = (src_idx, best_match.name)
        
        # 5. 교환 실행
        if best_swap_pair:
            s_idx, d_idx = best_swap_pair
            # 서로 반을 맞바꿈
            val_src = df.at[s_idx, '신학년반']
            val_dst = df.at[d_idx, '신학년반']
            
            df.at[s_idx, '신학년반'] = val_dst
            df.at[d_idx, '신학년반'] = val_src
        else:
            # 교환 대상이 없으면 루프 중단
            break
            
    return df

# --------------------------------------------------------------------------
# 3. 세션 및 메인 로직
# --------------------------------------------------------------------------
if 'df_result' not in st.session_state:
    st.session_state.df_result = None

uploaded_file = st.file_uploader("학생 성적 엑셀 파일을 업로드하세요 (.xlsx)", type=['xlsx', 'csv'])

if uploaded_file is not None and st.session_state.df_result is None:
    try:
        file_name = uploaded_file.name
        if file_name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
            
        df, error_msg = preprocess_data(df_raw)
        
        if error_msg:
            st.error(error_msg)
        else:
            # 1. 1차 배정 (순환 S자)
            df_allocated = allocate_class_logic(df)
            
            # 2. 2차 조정 (강력한 생활지도 분산)
            df_allocated = df_allocated.reset_index(drop=True)
            df_final = distribute_special_students_global(df_allocated)
            
            # 3. 비고 생성
            df_final['비고'] = df_final['생활지도_표시'].apply(lambda x: '★생활지도' if x else '')
            
            st.session_state.df_result = df_final
            st.success("✅ 반편성 완료! (생활지도 학생 균등 분산 적용됨)")
            st.rerun()

    except Exception as e:
        st.error(f"오류 발생: {e}")

# --------------------------------------------------------------------------
# 4. 결과 화면
# --------------------------------------------------------------------------
if st.session_state.df_result is not None:
    df_display = st.session_state.df_result.copy()
    
    # 정렬
    df_display['성별_order'] = df_display['성별'].apply(lambda x: 0 if x != '남' else 1)
    df_display = df_display.sort_values(by=['신학년반', '성별_order', '이름']).reset_index(drop=True)
    
    cols = ['신학년반', '이름', '성별', '2025반', '2025번호', '총점', '비고']
    
    # 다운로드 버튼
    col_h, col_b = st.columns([3, 1])
    with col_h: st.subheader("📋 반편성 결과")
    with col_b:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_display[cols].to_excel(writer, index=False, sheet_name='반편성결과')
        st.download_button("📥 엑셀 다운로드", data=output.getvalue(), file_name="2026_반편성_최종.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")

    st.divider()

    # 맞교환 기능
    with st.expander("🔄 학생 반 맞교환 (수동)", expanded=True):
        df_display['선택라벨'] = df_display.apply(lambda x: f"{x['이름']} ({x['신학년반']} / {x['총점']}점 / 구 {x['2025반']}반)", axis=1)
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1: s_a = st.selectbox("학생 A", df_display['선택라벨'].unique(), key='a')
        with c2: s_b = st.selectbox("학생 B", df_display['선택라벨'].unique(), key='b')
        with c3:
            st.write(""); st.write("")
            if st.button("교환하기"):
                if s_a != s_b:
                    # 원본 데이터 인덱스 찾기
                    try:
                        row_a = df_display[df_display['선택라벨'] == s_a].iloc[0]
                        row_b = df_display[df_display['선택라벨'] == s_b].iloc[0]
                        
                        # 이름, 반, 번호로 고유 식별
                        idx_a = st.session_state.df_result[
                            (st.session_state.df_result['이름'] == row_a['이름']) &
                            (st.session_state.df_result['2025반'] == row_a['2025반']) &
                            (st.session_state.df_result['2025번호'] == row_a['2025번호'])
                        ].index[0]
                        
                        idx_b = st.session_state.df_result[
                            (st.session_state.df_result['이름'] == row_b['이름']) &
                            (st.session_state.df_result['2025반'] == row_b['2025반']) &
                            (st.session_state.df_result['2025번호'] == row_b['2025번호'])
                        ].index[0]
                        
                        val_a = st.session_state.df_result.at[idx_a, '신학년반']
                        val_b = st.session_state.df_result.at[idx_b, '신학년반']
                        
                        st.session_state.df_result.at[idx_a, '신학년반'] = val_b
                        st.session_state.df_result.at[idx_b, '신학년반'] = val_a
                        
                        st.success("교환 완료!"); st.rerun()
                    except Exception as e:
                        st.error(f"오류가 발생했습니다: {e}")
                else:
                    st.warning("다른 학생을 선택해주세요.")

    # 탭 화면
    tabs = st.tabs(["가반", "나반", "다반", "전체"])
    
    def show_tab(cls_name):
        subset = df_display[df_display['신학년반'] == cls_name][cols]
        count = len(subset)
        special = len(subset[subset['비고'] != ''])
        avg = subset['총점'].mean() if count > 0 else 0
        
        msg = f"👥 총원: {count}명 | ⚠️ 생활지도: {special}명 | 📊 평균점수: {avg:.1f}점"
        
        # 4~6명 적정선 시각화
        if 4 <= special <= 6:
            st.success(msg + " (적정)")
        else:
            st.warning(msg + " (조정 권장)")
        
        st.dataframe(
            subset.style.apply(lambda x: ['background-color: #ffcccc' if v else '' for v in x], subset=['비고'], axis=1),
            use_container_width=True, hide_index=True, height=800
        )

    with tabs[0]: show_tab('가')
    with tabs[1]: show_tab('나')
    with tabs[2]: show_tab('다')
    with tabs[3]: st.dataframe(df_display[cols], use_container_width=True, height=800)
    
    if st.button("초기화"):
        st.session_state.df_result = None
        st.rerun()
