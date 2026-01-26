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
    /* 데이터프레임 헤더 스타일 */
    .stDataFrame { border: 1px solid #ddd; }
    </style>
""", unsafe_allow_html=True)

st.title("🏫 2026학년도 초등학교 반편성 시스템")
st.markdown("첨부된 **반편성 계획**에 의거하여 성적순 S자 배치 및 반 배정을 자동화합니다.")

# --------------------------------------------------------------------------
# 2. 데이터 처리 함수
# --------------------------------------------------------------------------
def preprocess_data(df):
    """데이터 정제 및 정수형 변환 (결측치 평균 대치 포함)"""
    # 1. 컬럼명 매핑
    col_map = {
        '성명': '이름',
        '합': '총점',
        '학반': '2025반',
        '번호': '2025번호',
        '생활지도 곤란': '생활지도'
    }
    df = df.rename(columns=col_map)
    
    # 2. 필수 컬럼 체크
    required = ['이름', '성별', '총점', '2025반', '2025번호']
    if not all(col in df.columns for col in required):
        return None, f"필수 컬럼이 누락되었습니다. (필요: {required}, 현재: {list(df.columns)})"

    # 3. 데이터 정제 (이름 없는 행 삭제)
    df = df.dropna(subset=['이름'])
    
    # 4. [수정됨] 점수 처리 로직
    # 일단 숫자로 변환 (에러나 빈값은 NaN으로 둠)
    df['총점'] = pd.to_numeric(df['총점'], errors='coerce')
    
    # 평균 계산 (NaN 제외한 나머지 학생들의 평균)
    avg_score = df['총점'].mean()
    if pd.isna(avg_score): avg_score = 0 # 데이터가 하나도 없으면 0점
    
    # 점수가 없는(NaN) 학생에게 평균 점수 부여
    df['총점'] = df['총점'].fillna(avg_score)
    
    # 정수 변환 (반올림)
    df['총점'] = df['총점'].round().astype(int)
    
    # 5. 반, 번호 정수 변환
    for col in ['2025반', '2025번호']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # 6. 생활지도 처리
    if '생활지도' in df.columns:
        df['생활지도_표시'] = df['생활지도'].notna() & (df['생활지도'] != 0)
    else:
        df['생활지도_표시'] = False
        
    return df, None

def s_shape_grouping(df):
    """성적순 S자 그룹핑"""
    df = df.sort_values(by=['총점', '이름'], ascending=[False, True]).reset_index(drop=True)
    groups = []
    for i in range(len(df)):
        cycle = i % 6 
        if cycle == 0: group = 'A'
        elif cycle == 1: group = 'B'
        elif cycle == 2: group = 'C'
        elif cycle == 3: group = 'C'
        elif cycle == 4: group = 'B'
        else: group = 'A'
        groups.append(group)
    df['그룹'] = groups
    return df

def assign_new_class(row):
    """구학년 반 -> 신학년 반 매핑"""
    old_class = str(row['2025반'])
    group = row['그룹']
    
    if old_class == '1':
        return {'A': '가', 'B': '다', 'C': '나'}.get(group, '미배정')
    elif old_class == '2':
        return {'A': '나', 'B': '가', 'C': '다'}.get(group, '미배정')
    elif old_class == '3':
        return {'A': '다', 'B': '나', 'C': '가'}.get(group, '미배정')
    return "미배정"

# --------------------------------------------------------------------------
# 3. 세션 상태 관리
# --------------------------------------------------------------------------
if 'df_result' not in st.session_state:
    st.session_state.df_result = None

# --------------------------------------------------------------------------
# 4. 메인 앱 로직
# --------------------------------------------------------------------------
uploaded_file = st.file_uploader("학생 성적 엑셀 파일을 업로드하세요 (.xlsx)", type=['xlsx', 'csv'])

# 파일이 업로드되었고, 아직 처리된 데이터가 세션에 없다면 최초 1회 실행
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
            # 1. 자동 편성 로직 수행
            df_male = df[df['성별'] == '남'].copy()
            df_female = df[df['성별'] != '남'].copy()
            
            df_male = s_shape_grouping(df_male)
            df_female = s_shape_grouping(df_female)
            
            result_df = pd.concat([df_male, df_female], ignore_index=True)
            result_df['신학년반'] = result_df.apply(assign_new_class, axis=1)
            
            # 2. 비고란 생성
            result_df['비고'] = result_df['생활지도_표시'].apply(lambda x: '★생활지도(3점)' if x else '')
            
            # 3. 세션에 저장
            st.session_state.df_result = result_df
            st.success("✅ 자동 반편성이 완료되었습니다. (점수 미기재 학생은 평균 점수로 자동 적용됨)")
            st.rerun()

    except Exception as e:
        st.error(f"오류 발생: {e}")

# --------------------------------------------------------------------------
# 5. 결과 화면 (수정 및 조회)
# --------------------------------------------------------------------------
if st.session_state.df_result is not None:
    df_display = st.session_state.df_result.copy()
    
    # 정렬 (화면 표시용)
    df_display['성별_order'] = df_display['성별'].apply(lambda x: 0 if x != '남' else 1)
    df_display = df_display.sort_values(by=['신학년반', '성별_order', '이름']).reset_index(drop=True)
    
    # 표시할 컬럼 정의
    cols = ['신학년반', '이름', '성별', '2025반', '2025번호', '총점', '그룹', '비고']
    
    # [상단] 엑셀 다운로드 버튼
    col_header, col_btn = st.columns([3, 1])
    with col_header:
        st.subheader("📋 반편성 결과 확인 및 수정")
    with col_btn:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_display[cols].to_excel(writer, index=False, sheet_name='반편성결과')
        st.download_button(
            label="📥 결과 엑셀 다운로드",
            data=output.getvalue(),
            file_name="2026_반편성_최종.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )

    st.divider()

    # [중단] 학생 맞교환 기능
    with st.expander("🔄 학생 반 맞교환하기 (수동 조정)", expanded=True):
        st.info("생활지도 문제나 교우관계를 고려하여 두 학생의 반을 서로 맞바꿀 수 있습니다.")
        
        # 선택박스용 라벨 생성
        df_display['선택라벨'] = df_display.apply(
            lambda x: f"{x['이름']} ({x['신학년반']} / 구 {x['2025반']}반)", axis=1
        )
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            student_a_label = st.selectbox("학생 A 선택", df_display['선택라벨'].unique(), key='std_a')
        with col2:
            student_b_label = st.selectbox("학생 B 선택", df_display['선택라벨'].unique(), key='std_b')
        with col3:
            st.write("") # UI 간격 조절
            st.write("")
            if st.button("🔄 서로 맞바꾸기"):
                if student_a_label == student_b_label:
                    st.warning("서로 다른 학생을 선택해주세요.")
                else:
                    # 선택된 학생의 행 찾기
                    row_a = df_display[df_display['선택라벨'] == student_a_label].iloc[0]
                    row_b = df_display[df_display['선택라벨'] == student_b_label].iloc[0]
                    
                    # 원본 데이터(session_state)에서 인덱스 찾기
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
                    
                    # 반 교체
                    class_a = st.session_state.df_result.at[idx_a, '신학년반']
                    class_b = st.session_state.df_result.at[idx_b, '신학년반']
                    
                    st.session_state.df_result.at[idx_a, '신학년반'] = class_b
                    st.session_state.df_result.at[idx_b, '신학년반'] = class_a
                    
                    st.success(f"{row_a['이름']} 학생과 {row_b['이름']} 학생의 반이 교체되었습니다.")
                    st.rerun()

    # [하단] 탭별 조회 (화면 길이 800px로 확장)
    tabs = st.tabs(["가반", "나반", "다반", "전체 명부"])
    
    def show_class_table(class_name):
        subset = df_display[df_display['신학년반'] == class_name][cols]
        
        target_count = len(subset[subset['비고'] != ''])
        if target_count > 0:
            st.warning(f"⚠️ 이 반에는 생활지도 고려 학생이 {target_count}명 포함되어 있습니다.")
            
        st.dataframe(
            subset.style.apply(lambda x: ['background-color: #ffcccc' if v != '' else '' for v in x], subset=['비고'], axis=1),
            use_container_width=True,
            hide_index=True,
            height=800 
        )
    
    with tabs[0]: show_class_table('가')
    with tabs[1]: show_class_table('나')
    with tabs[2]: show_class_table('다')
    with tabs[3]: 
        st.dataframe(df_display[cols], use_container_width=True, height=800, hide_index=True)

    # 초기화 버튼
    if st.button("초기화 (새 파일 업로드)"):
        st.session_state.df_result = None
        st.rerun()
