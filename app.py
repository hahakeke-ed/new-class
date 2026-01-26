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
st.markdown("첨부된 **반편성 계획**에 의거하여 **각 반별 균등 분할** 및 **생활지도 학생 분산**을 수행합니다.")

# --------------------------------------------------------------------------
# 2. 데이터 처리 및 알고리즘 함수
# --------------------------------------------------------------------------
def preprocess_data(df):
    """데이터 정제 (텍스트 인식 포함)"""
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
    
    # 점수 처리 (평균 대치)
    df['총점'] = pd.to_numeric(df['총점'], errors='coerce')
    avg_score = df['총점'].mean()
    if pd.isna(avg_score): avg_score = 0 
    df['총점'] = df['총점'].fillna(avg_score).round().astype(int)
    
    # 반, 번호 정수 처리
    for col in ['2025반', '2025번호']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # [생활지도] 텍스트나 숫자가 있으면 True
    if '생활지도' in df.columns:
        df['생활지도_표시'] = df['생활지도'].astype(str).apply(
            lambda x: True if x.strip() not in ['nan', '', '0', '0.0', 'None'] else False
        )
    else:
        df['생활지도_표시'] = False
        
    return df, None

def s_shape_grouping_logic(subset_df):
    """
    S자 그룹핑 로직 (단일 그룹용)
    """
    # 성적순 정렬 (동점자는 이름순)
    subset_df = subset_df.sort_values(by=['총점', '이름'], ascending=[False, True]).reset_index(drop=True)
    groups = []
    
    # 6명 단위 S자 패턴 (A->B->C->C->B->A)
    for i in range(len(subset_df)):
        cycle = i % 6 
        if cycle == 0: group = 'A'
        elif cycle == 1: group = 'B'
        elif cycle == 2: group = 'C'
        elif cycle == 3: group = 'C'
        elif cycle == 4: group = 'B'
        else: group = 'A'
        groups.append(group)
        
    subset_df['그룹'] = groups
    # 성적 순위 저장 (추후 교환 로직에서 사용)
    subset_df['성적순위'] = subset_df['총점'].rank(method='min', ascending=False)
    return subset_df

def apply_grouping_by_class(df):
    """
    [핵심 수정] 전체가 아니라 '각 반별 + 성별'로 나누어 그룹핑 수행
    이렇게 해야 1반 안에서 A,B,C가 1:1:1로 나오고, 결과적으로 전체 인원 균형이 맞음
    """
    grouped_results = []
    
    # 2025반과 성별로 그룹을 나눔 (예: 1반 남, 1반 여, 2반 남...)
    # groupby 객체를 리스트로 변환하지 않고 직접 순회
    for (cls, gender), group_df in df.groupby(['2025반', '성별']):
        processed_group = s_shape_grouping_logic(group_df.copy())
        grouped_results.append(processed_group)
        
    if not grouped_results:
        return df # 데이터가 없을 경우
        
    return pd.concat(grouped_results, ignore_index=True)

def assign_new_class(row):
    """기본 반 배정 로직"""
    old_class = str(row['2025반'])
    group = row['그룹']
    
    # 2025반 데이터가 1,2,3 외의 숫자일 경우 처리 필요하지만
    # 기본적으로 1,2,3반 로직만 문서에 있으므로 이에 따름
    if old_class == '1':
        return {'A': '가', 'B': '다', 'C': '나'}.get(group, '미배정')
    elif old_class == '2':
        return {'A': '나', 'B': '가', 'C': '다'}.get(group, '미배정')
    elif old_class == '3':
        return {'A': '다', 'B': '나', 'C': '가'}.get(group, '미배정')
    return "미배정"

def distribute_special_students(df):
    """
    생활지도 학생 자동 분산 (1:1 교환 방식이라 인원수 변화 없음)
    """
    max_iter = 10 # 반복 횟수 증가
    
    for _ in range(max_iter):
        # 전체 반별 생활지도 학생 수 체크
        counts = df[df['생활지도_표시'] == True]['신학년반'].value_counts()
        if counts.empty: break
        
        max_count = counts.max()
        min_count = counts.min()
        
        # 차이가 1명 이하면 균형으로 간주
        if max_count - min_count <= 1:
            break
            
        overloaded_class = counts.idxmax()
        
        # 가장 적은 반 찾기 (가,나,다 중)
        all_classes = ['가', '나', '다']
        current_counts = {c: counts.get(c, 0) for c in all_classes}
        target_class = min(current_counts, key=current_counts.get)
        
        # 교환 대상 1: 과밀 반의 생활지도 학생
        candidates = df[(df['신학년반'] == overloaded_class) & (df['생활지도_표시'] == True)]
        if candidates.empty: break
        
        target_student = candidates.iloc[0]
        target_idx = target_student.name 
        
        # 교환 대상 2: 부족 반의 일반 학생 (성별 같아야 함!)
        # 성별 조건을 추가하여 남녀 성비 유지
        target_gender = target_student['성별']
        
        dest_candidates = df[
            (df['신학년반'] == target_class) & 
            (df['생활지도_표시'] == False) &
            (df['성별'] == target_gender)
        ].copy()
        
        if dest_candidates.empty: break 
        
        # 성적 차이가 가장 적은 학생 찾기
        dest_candidates['score_diff'] = abs(dest_candidates['총점'] - target_student['총점'])
        swap_student = dest_candidates.sort_values('score_diff').iloc[0]
        swap_idx = swap_student.name
        
        # 맞교환
        df.at[target_idx, '신학년반'] = target_class
        df.at[swap_idx, '신학년반'] = overloaded_class
        
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
            # 1. [핵심 변경] 전체가 아닌 '반별+성별' 그룹핑
            # 이렇게 하면 1반 남학생 10명이면 A,B,C가 3,4,3명으로 나뉨 -> 인원 균형 보장
            df_grouped = apply_grouping_by_class(df)
            
            # 2. 반 배정 (미배정 데이터 필터링 가능성 대비)
            df_grouped['신학년반'] = df_grouped.apply(assign_new_class, axis=1)
            
            # 미배정(4반 등)이 있을 수 있으므로 가/나/다 만 필터링하거나 그대로 둠
            # 여기서는 로직상 '미배정' 텍스트가 들어갈 수 있음
            
            # 3. 생활지도 학생 분산 (성별 내부 교환이므로 인원/성비 불변)
            # 남/녀 각각 최적화 수행
            mask_male = df_grouped['성별'] == '남'
            df_m_opt = distribute_special_students(df_grouped[mask_male].copy())
            
            mask_female = df_grouped['성별'] != '남'
            df_f_opt = distribute_special_students(df_grouped[mask_female].copy())
            
            # 인덱스 기준으로 원본 업데이트
            df_grouped.update(df_m_opt)
            df_grouped.update(df_f_opt)
            
            # 4. 비고 생성
            df_grouped['비고'] = df_grouped['생활지도_표시'].apply(lambda x: '★생활지도' if x else '')
            
            st.session_state.df_result = df_grouped
            st.success("✅ 반편성 완료! (각 반별 인원 균등 배분 적용됨)")
            st.rerun()

    except Exception as e:
        st.error(f"오류 발생: {e}")

# --------------------------------------------------------------------------
# 4. 결과 화면
# --------------------------------------------------------------------------
if st.session_state.df_result is not None:
    df_display = st.session_state.df_result.copy()
    
    # 정렬 (가나다 -> 성별(여우선) -> 이름)
    df_display['성별_order'] = df_display['성별'].apply(lambda x: 0 if x != '남' else 1)
    df_display = df_display.sort_values(by=['신학년반', '성별_order', '이름']).reset_index(drop=True)
    
    cols = ['신학년반', '이름', '성별', '2025반', '2025번호', '총점', '그룹', '비고']
    
    # 다운로드 버튼
    col_h, col_b = st.columns([3, 1])
    with col_h: st.subheader("📋 반편성 결과")
    with col_b:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_display[cols].to_excel(writer, index=False, sheet_name='반편성결과')
        st.download_button("📥 엑셀 다운로드", data=output.getvalue(), file_name="2026_반편성_최종.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")

    st.divider()

    # 맞교환 UI
    with st.expander("🔄 학생 반 맞교환 (수동)", expanded=True):
        df_display['선택라벨'] = df_display.apply(lambda x: f"{x['이름']} ({x['신학년반']} / {x['총점']}점 / 구 {x['2025반']}반)", axis=1)
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1: s_a = st.selectbox("학생 A", df_display['선택라벨'].unique(), key='a')
        with c2: s_b = st.selectbox("학생 B", df_display['선택라벨'].unique(), key='b')
        with c3:
            st.write(""); st.write("")
            if st.button("교환하기"):
                if s_a != s_b:
                    def get_idx(label):
                        r = df_display[df_display['선택라벨'] == label].iloc[0]
                        return st.session_state.df_result[
                            (st.session_state.df_result['이름'] == r['이름']) &
                            (st.session_state.df_result['2025반'] == r['2025반']) &
                            (st.session_state.df_result['2025번호'] == r['2025번호'])
                        ].index[0]
                    
                    idx_a = get_idx(s_a)
                    idx_b = get_idx(s_b)
                    
                    val_a = st.session_state.df_result.at[idx_a, '신학년반']
                    val_b = st.session_state.df_result.at[idx_b, '신학년반']
                    
                    st.session_state.df_result.at[idx_a, '신학년반'] = val_b
                    st.session_state.df_result.at[idx_b, '신학년반'] = val_a
                    st.success("교환 완료!"); st.rerun()

    # 결과 탭
    tabs = st.tabs(["가반", "나반", "다반", "전체"])
    
    def show_tab(cls_name):
        subset = df_display[df_display['신학년반'] == cls_name][cols]
        count = len(subset)
        special = len(subset[subset['비고'] != ''])
        avg = subset['총점'].mean() if count > 0 else 0
        
        st.info(f"👥 총원: {count}명 | ⚠️ 생활지도: {special}명 | 📊 평균점수: {avg:.1f}점")
        
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
