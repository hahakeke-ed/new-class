import streamlit as st
import pandas as pd
import io
import math

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
st.markdown("첨부된 **반편성 계획**에 의거하여 성적순 S자 배치 및 **생활지도 학생 자동 분산**을 수행합니다.")

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
    
    # [핵심] 생활지도 컬럼이 비어있지 않으면(숫자든 텍스트든) True
    if '생활지도' in df.columns:
        # 빈 문자열이나 NaN이 아닌 경우 모두 True로 처리
        df['생활지도_표시'] = df['생활지도'].astype(str).apply(lambda x: True if x.strip() not in ['nan', '', '0', '0.0'] else False)
    else:
        df['생활지도_표시'] = False
        
    return df, None

def s_shape_grouping(df):
    """
    성적순 S자 그룹핑 (인원 불균형 자동 해소)
    나머지가 생기면(예: 29, 29, 28) 앞에서부터 채워지므로 자연스럽게 배분됨
    """
    df = df.sort_values(by=['총점', '이름'], ascending=[False, True]).reset_index(drop=True)
    groups = []
    # 전체 인원을 돌면서 6개 단위 패턴 반복 -> 남는 인원은 순서대로 A, B... 배정됨
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
    
    # 나중에 교환 로직을 위해 '석차' 정보를 저장해둠 (성적 유사도 판단용)
    df['성적순위'] = df.index
    return df

def assign_new_class(row):
    """기본 반 배정 로직"""
    old_class = str(row['2025반'])
    group = row['그룹']
    
    if old_class == '1':
        return {'A': '가', 'B': '다', 'C': '나'}.get(group, '미배정')
    elif old_class == '2':
        return {'A': '나', 'B': '가', 'C': '다'}.get(group, '미배정')
    elif old_class == '3':
        return {'A': '다', 'B': '나', 'C': '가'}.get(group, '미배정')
    return "미배정"

def distribute_special_students(df):
    """
    [핵심 기능] 생활지도 학생 자동 분산 알고리즘
    - 반별 생활지도 학생 수를 체크하여, 몰려있으면 다른 반의 '비슷한 등수' 학생과 맞교환
    """
    # 최대 반복 횟수 (무한 루프 방지)
    max_iter = 5
    
    for _ in range(max_iter):
        # 현재 반별 생활지도 학생 수 집계
        counts = df[df['생활지도_표시'] == True]['신학년반'].value_counts()
        if counts.empty: break
        
        max_count = counts.max()
        min_count = counts.min()
        
        # 가장 많은 반과 적은 반의 차이가 1명 이하면 균형 잡힌 것으로 간주 (종료)
        if max_count - min_count <= 1:
            break
            
        # 과밀 학급과 부족 학급 식별
        overloaded_class = counts.idxmax()
        # 부족한 반 찾기 (가, 나, 다 중 counts에 없거나 가장 적은 반)
        all_classes = ['가', '나', '다']
        current_counts = {c: counts.get(c, 0) for c in all_classes}
        target_class = min(current_counts, key=current_counts.get)
        
        # 교환 대상 찾기 (과밀 반의 생활지도 학생 중 하나)
        # 성적 순위를 기준으로 정렬해서, 가능한 중간 등수의 학생을 옮기는 게 안전하지만
        # 여기서는 가장 먼저 발견된 학생을 이동 시도
        candidates = df[(df['신학년반'] == overloaded_class) & (df['생활지도_표시'] == True)]
        
        if candidates.empty: break
        
        # 이동할 생활지도 학생 (Target A)
        target_student = candidates.iloc[0]
        target_idx = target_student.name # DataFrame Index
        target_rank = target_student['성적순위']
        
        # 맞교환할 상대방 찾기 (Target B: 부족 반의 일반 학생 중 성적이 가장 비슷한 학생)
        # 조건: 생활지도가 아니어야 함
        dest_candidates = df[(df['신학년반'] == target_class) & (df['생활지도_표시'] == False)].copy()
        
        if dest_candidates.empty: break # 교환할 일반 학생이 없으면 중단
        
        # 성적 순위 차이가 가장 적은 학생 찾기
        dest_candidates['rank_diff'] = abs(dest_candidates['성적순위'] - target_rank)
        swap_student = dest_candidates.sort_values('rank_diff').iloc[0]
        swap_idx = swap_student.name
        
        # 맞교환 실행
        df.at[target_idx, '신학년반'] = target_class
        df.at[swap_idx, '신학년반'] = overloaded_class
        
        # 루프 다시 돌면서 균형 맞을 때까지 반복
        
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
            # 1. 성별 분리 후 S자 그룹핑 (성적 균형)
            df_male = df[df['성별'] == '남'].copy()
            df_female = df[df['성별'] != '남'].copy()
            
            df_male = s_shape_grouping(df_male)
            df_female = s_shape_grouping(df_female)
            
            # 2. 합치기 (인덱스 리셋하여 고유 ID처럼 사용)
            result_df = pd.concat([df_male, df_female], ignore_index=True)
            
            # 3. 1차 반 배정
            result_df['신학년반'] = result_df.apply(assign_new_class, axis=1)
            
            # 4. [New] 생활지도 학생 자동 분산 (남/녀 각각 수행하여 성비 유지)
            # 남자 안에서 교환, 여자 안에서 교환해야 성비가 안 깨짐
            
            # 남자 분산
            mask_male = result_df['성별'] == '남'
            df_m_only = result_df[mask_male].copy()
            df_m_optimized = distribute_special_students(df_m_only)
            result_df.update(df_m_optimized)
            
            # 여자 분산
            mask_female = result_df['성별'] != '남'
            df_f_only = result_df[mask_female].copy()
            df_f_optimized = distribute_special_students(df_f_only)
            result_df.update(df_f_optimized)
            
            # 5. 비고 및 최종 정리
            result_df['비고'] = result_df['생활지도_표시'].apply(lambda x: '★생활지도' if x else '')
            
            st.session_state.df_result = result_df
            st.success("✅ 자동 반편성 완료! (인원 균형 및 생활지도 학생 분산 적용됨)")
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
    
    cols = ['신학년반', '이름', '성별', '2025반', '2025번호', '총점', '그룹', '비고']
    
    # 상단 다운로드 버튼
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
        df_display['선택라벨'] = df_display.apply(lambda x: f"{x['이름']} ({x['신학년반']} / {x['총점']}점)", axis=1)
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1: s_a = st.selectbox("학생 A", df_display['선택라벨'].unique(), key='a')
        with c2: s_b = st.selectbox("학생 B", df_display['선택라벨'].unique(), key='b')
        with c3:
            st.write(""); st.write("")
            if st.button("교환하기"):
                if s_a != s_b:
                    # 원본 인덱스 찾아서 교환 (이름, 2025반, 번호 기준)
                    def get_real_idx(label):
                        row = df_display[df_display['선택라벨'] == label].iloc[0]
                        return st.session_state.df_result[
                            (st.session_state.df_result['이름'] == row['이름']) &
                            (st.session_state.df_result['2025반'] == row['2025반']) &
                            (st.session_state.df_result['2025번호'] == row['2025번호'])
                        ].index[0]
                    
                    idx_a = get_real_idx(s_a)
                    idx_b = get_real_idx(s_b)
                    
                    val_a = st.session_state.df_result.at[idx_a, '신학년반']
                    val_b = st.session_state.df_result.at[idx_b, '신학년반']
                    
                    st.session_state.df_result.at[idx_a, '신학년반'] = val_b
                    st.session_state.df_result.at[idx_b, '신학년반'] = val_a
                    st.success("교환 완료!"); st.rerun()

    # 탭 화면 (요약 정보 포함)
    tabs = st.tabs(["가반", "나반", "다반", "전체"])
    
    def show_tab(cls_name):
        subset = df_display[df_display['신학년반'] == cls_name][cols]
        count = len(subset)
        special_count = len(subset[subset['비고'] != ''])
        avg = subset['총점'].mean()
        
        st.info(f"👥 총원: {count}명 | ⚠️ 생활지도: {special_count}명 | 📊 평균점수: {avg:.1f}점")
        
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
