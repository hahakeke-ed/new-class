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
1. **학반별 순환 배정:** 1반(가→나→다), 2반(나→다→가)... 순환 S자 배정
2. **생활지도 분산:** 반별 생활지도 학생 수 균형 (4~6명 목표)
3. **[New] 출신 학반 안배:** 각 반에 구학년(2025) 같은 반 친구가 **최소 4명 이상** 포함되도록 유지
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
    
    # 생활지도 및 분리 대상 처리
    if '생활지도' in df.columns:
        df['생활지도_표시'] = df['생활지도'].astype(str).apply(
            lambda x: True if x.strip() not in ['nan', '', '0', '0.0', 'None'] else False
        )
        def extract_conflict(val):
            s = str(val).strip()
            if s in ['nan', '', 'None', '0', '0.0']: return None
            if not s.replace('.', '').isdigit(): return s
            return None
        df['분리대상'] = df['생활지도'].apply(extract_conflict)
    else:
        df['생활지도_표시'] = False
        df['분리대상'] = None
        
    return df, None

def check_conflict_safety(df, student_idx, target_class):
    """분리 배정(앙숙) 안전 체크"""
    student = df.loc[student_idx]
    enemy_name = student['분리대상']
    
    # 내가 피해야 할 사람이 저기에 있는가?
    if enemy_name:
        enemies = df[(df['이름'] == enemy_name) & (df['신학년반'] == target_class)]
        if not enemies.empty: return False

    # 저기에 있는 누군가가 나를 피해야 하는가?
    my_name = student['이름']
    haters = df[(df['신학년반'] == target_class) & (df['분리대상'] == my_name)]
    if not haters.empty: return False
        
    return True

def check_old_class_constraint(df, student_idx, current_class, min_count=4):
    """
    [New] 구학년 반 친구 최소 인원(4명) 보존 체크
    이 학생이 빠져나가도, 해당 반에 구학년 친구들이 min_count 이상 남는가?
    """
    student = df.loc[student_idx]
    old_cls = student['2025반']
    
    # 현재 반에 있는 '나와 같은 구반 친구들' 수 (나 포함)
    count = len(df[
        (df['신학년반'] == current_class) & 
        (df['2025반'] == old_cls)
    ])
    
    # 내가 나가면 (count - 1)명이 됨.
    # 그 수가 min_count보다 작아지면 안 됨. (즉, 현재 count가 min_count보다 커야 나갈 수 있음)
    if count <= min_count:
        return False # 이동 불가 (인원 부족)
    
    return True # 이동 가능

def allocate_class_logic(df):
    """초기 배정 (순환 S자)"""
    results = []
    for (old_class, gender), sub_df in df.groupby(['2025반', '성별']):
        sub_df = sub_df.sort_values(by=['총점', '이름'], ascending=[False, True]).copy()
        
        if old_class == 1: targets = ['가', '나', '다']
        elif old_class == 2: targets = ['나', '다', '가']
        elif old_class == 3: targets = ['다', '가', '나']
        else: targets = ['가', '나', '다']
            
        new_classes = []
        for i in range(len(sub_df)):
            idx = [0, 1, 2, 2, 1, 0][i % 6]
            new_classes.append(targets[idx])
            
        sub_df['신학년반'] = new_classes
        results.append(sub_df)
        
    if not results: return df
    return pd.concat(results, ignore_index=True)

def resolve_conflicts_first(df):
    """1단계: 분리 배정 해결 (최소 인원 조건 준수 포함)"""
    classes = ['가', '나', '다']
    conflict_rows = df[df['분리대상'].notna()]
    
    for idx, row in conflict_rows.iterrows():
        enemy_name = row['분리대상']
        my_class = row['신학년반']
        
        enemies = df[(df['이름'] == enemy_name) & (df['신학년반'] == my_class)]
        
        if not enemies.empty:
            # 이동 필요. 이동 가능한 반 찾기
            available_classes = [c for c in classes if c != my_class]
            
            for target_class in available_classes:
                # 1. 앙숙 안전 체크
                if not check_conflict_safety(df, idx, target_class): continue
                
                # 2. [New] 내 구반 친구들이 4명 이하로 떨어지면 못 나감
                if not check_old_class_constraint(df, idx, my_class): continue
                
                # 교환 대상 찾기
                swap_candidates = df[
                    (df['신학년반'] == target_class) &
                    (df['생활지도_표시'] == False) &
                    (df['성별'] == row['성별'])
                ]
                
                # 교환 대상의 조건 체크
                valid_swap_target = None
                
                # 점수 차이 순으로 정렬하여 시도
                swap_candidates = swap_candidates.copy()
                swap_candidates['diff'] = abs(swap_candidates['총점'] - row['총점'])
                swap_candidates = swap_candidates.sort_values('diff')
                
                for s_idx, s_row in swap_candidates.iterrows():
                    # 대상이 내 반으로 와도 안전한가?
                    if not check_conflict_safety(df, s_idx, my_class): continue
                    
                    # [New] 대상이 자기 반을 떠나도 구반 친구가 유지되는가?
                    if not check_old_class_constraint(df, s_idx, target_class): continue
                    
                    valid_swap_target = s_idx
                    break
                
                if valid_swap_target:
                    df.at[idx, '신학년반'] = target_class
                    df.at[valid_swap_target, '신학년반'] = my_class
                    break 
    return df

def distribute_special_students_global(df):
    """2단계: 생활지도 균형 (최소 인원 조건 준수 포함)"""
    max_iter = 300
    
    for i in range(max_iter):
        counts = df[df['생활지도_표시'] == True]['신학년반'].value_counts()
        for cls in ['가', '나', '다']:
            if cls not in counts: counts[cls] = 0
            
        if counts.max() - counts.min() <= 1:
            break
            
        src_class = counts.idxmax()
        dst_class = counts.idxmin()
        
        src_candidates = df[(df['신학년반'] == src_class) & (df['생활지도_표시'] == True)]
        
        best_swap_pair = None
        min_score_diff = float('inf')
        
        for src_idx, src_student in src_candidates.iterrows():
            # 안전 및 제약 조건 체크
            if not check_conflict_safety(df, src_idx, dst_class): continue
            if not check_old_class_constraint(df, src_idx, src_class): continue # [New]
            
            s_gender = src_student['성별']
            s_score = src_student['총점']
            
            dst_candidates = df[
                (df['신학년반'] == dst_class) & 
                (df['생활지도_표시'] == False) & 
                (df['성별'] == s_gender)
            ]
            
            for d_idx, d_row in dst_candidates.iterrows():
                if not check_conflict_safety(df, d_idx, src_class): continue
                if not check_old_class_constraint(df, d_idx, dst_class): continue # [New]
                
                diff = abs(d_row['총점'] - s_score)
                if diff < min_score_diff:
                    min_score_diff = diff
                    best_swap_pair = (src_idx, d_idx)
        
        if best_swap_pair:
            s_idx, d_idx = best_swap_pair
            val_src = df.at[s_idx, '신학년반']
            val_dst = df.at[d_idx, '신학년반']
            df.at[s_idx, '신학년반'] = val_dst
            df.at[d_idx, '신학년반'] = val_src
        else:
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
            # 1. 초기 배정
            df_allocated = allocate_class_logic(df)
            df_allocated = df_allocated.reset_index(drop=True)
            
            # 2. 앙숙 관계 해결 (제약 조건 포함)
            df_allocated = resolve_conflicts_first(df_allocated)
            
            # 3. 생활지도 균형 (제약 조건 포함)
            df_final = distribute_special_students_global(df_allocated)
            
            # 비고 생성
            def make_note(row):
                notes = []
                if row['생활지도_표시']: notes.append('★생활지도')
                if row['분리대상']: notes.append(f"(분리:{row['분리대상']})")
                return ' '.join(notes)
            df_final['비고'] = df_final.apply(make_note, axis=1)
            
            st.session_state.df_result = df_final
            st.success("✅ 반편성 완료! (출신 학반 4명 이상 보장 & 생활지도 분산)")
            st.rerun()

    except Exception as e:
        st.error(f"오류 발생: {e}")

# --------------------------------------------------------------------------
# 4. 결과 화면
# --------------------------------------------------------------------------
if st.session_state.df_result is not None:
    df_display = st.session_state.df_result.copy()
    
    df_display['성별_order'] = df_display['성별'].apply(lambda x: 0 if x != '남' else 1)
    df_display = df_display.sort_values(by=['신학년반', '성별_order', '이름']).reset_index(drop=True)
    
    cols = ['신학년반', '이름', '성별', '2025반', '2025번호', '총점', '비고']
    
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
                    try:
                        row_a = df_display[df_display['선택라벨'] == s_a].iloc[0]
                        row_b = df_display[df_display['선택라벨'] == s_b].iloc[0]
                        
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
                        
                        # 안전 체크
                        safe_a = check_conflict_safety(st.session_state.df_result, idx_a, row_b['신학년반'])
                        safe_b = check_conflict_safety(st.session_state.df_result, idx_b, row_a['신학년반'])
                        
                        if not (safe_a and safe_b):
                            st.warning("⚠️ 주의: 분리 배정 원칙 위반 가능성이 있습니다.")
                            
                        # 구반 인원 체크 (수동은 경고만 하고 허용)
                        valid_old_a = check_old_class_constraint(st.session_state.df_result, idx_a, row_a['신학년반'])
                        valid_old_b = check_old_class_constraint(st.session_state.df_result, idx_b, row_b['신학년반'])
                        
                        if not (valid_old_a and valid_old_b):
                            st.warning("⚠️ 주의: 구학년 친구 최소 인원(4명)이 깨질 수 있습니다.")

                        val_a = st.session_state.df_result.at[idx_a, '신학년반']
                        val_b = st.session_state.df_result.at[idx_b, '신학년반']
                        st.session_state.df_result.at[idx_a, '신학년반'] = val_b
                        st.session_state.df_result.at[idx_b, '신학년반'] = val_a
                        
                        st.success("교환 완료!"); st.rerun()
                    except: st.error("오류 발생")
                else: st.warning("다른 학생 선택")

    tabs = st.tabs(["가반", "나반", "다반", "전체"])
    
    def show_tab(cls_name):
        subset = df_display[df_display['신학년반'] == cls_name][cols]
        count = len(subset)
        special = len(subset[subset['비고'].str.contains('생활지도')])
        avg = subset['총점'].mean() if count > 0 else 0
        
        # 구학년 분포 확인
        old_dist = subset['2025반'].value_counts().sort_index()
        old_dist_str = ", ".join([f"{k}반:{v}명" for k, v in old_dist.items()])
        
        st.info(f"👥 총원: {count}명 | ⚠️ 생활지도: {special}명 | 📊 평균: {avg:.1f}점")
        # 구반 분포 표시 (중요)
        if any(v < 4 for v in old_dist.values):
            st.error(f"🚨 출신 학반 분포 불균형: {old_dist_str} (4명 미만 있음!)")
        else:
            st.success(f"✅ 출신 학반 분포: {old_dist_str}")
        
        st.dataframe(
            subset.style.apply(lambda x: ['background-color: #ffcccc' if '생활지도' in v else '' for v in x], subset=['비고'], axis=1),
            use_container_width=True, hide_index=True, height=800
        )

    with tabs[0]: show_tab('가')
    with tabs[1]: show_tab('나')
    with tabs[2]: show_tab('다')
    with tabs[3]: st.dataframe(df_display[cols], use_container_width=True, height=800)
    
    if st.button("초기화"):
        st.session_state.df_result = None
        st.rerun()
