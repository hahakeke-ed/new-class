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
1. **학반별 순환 배정:** 1반(가→나→다), 2반(나→다→가), 3반(다→가→나) 로테이션
2. **S자형 성적 안배:** 성적 편차 최소화를 위해 S자(ㄹ자) 패턴 적용
3. **분리 배정 우선:** '생활지도'란에 이름이 적힌 경우, 해당 학생과 **절대 같은 반에 배치하지 않음**
4. **생활지도 균형:** 위 원칙을 지키면서 반별 생활지도 학생 수 균형(4~6명) 유지
""")

# --------------------------------------------------------------------------
# 2. 데이터 처리 및 알고리즘 함수
# --------------------------------------------------------------------------
def preprocess_data(df):
    """데이터 정제 및 분리 대상 파악"""
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
        # 1. 텍스트나 숫자가 있으면 생활지도 대상
        df['생활지도_표시'] = df['생활지도'].astype(str).apply(
            lambda x: True if x.strip() not in ['nan', '', '0', '0.0', 'None'] else False
        )
        # 2. 텍스트가 이름인 경우 분리 대상으로 저장
        # (숫자가 아니고 길이가 2 이상인 경우 이름으로 간주)
        def extract_conflict(val):
            s = str(val).strip()
            if s in ['nan', '', 'None', '0', '0.0']: return None
            # 숫자로만 구성된 게 아니라면(이름이라면) 반환
            if not s.replace('.', '').isdigit():
                return s
            return None
            
        df['분리대상'] = df['생활지도'].apply(extract_conflict)
    else:
        df['생활지도_표시'] = False
        df['분리대상'] = None
        
    return df, None

def check_conflict_safety(df, student_idx, target_class):
    """
    [안전 장치] student_idx 학생을 target_class로 보내도 안전한가?
    (target_class에 앙숙이 없는가?)
    """
    student = df.loc[student_idx]
    enemy_name = student['분리대상']
    
    # 1. 내가 피해야 할 학생이 target_class에 있는가?
    if enemy_name:
        # 이름으로 대상 찾기 (동명이인 고려 없이 이름 매칭)
        enemies = df[
            (df['이름'] == enemy_name) & 
            (df['신학년반'] == target_class)
        ]
        if not enemies.empty:
            return False # 위험!

    # 2. target_class에 있는 누군가가 나를 피해야 하는가?
    # (반대 방향 확인: 다른 학생의 '분리대상'이 나인 경우)
    my_name = student['이름']
    haters = df[
        (df['신학년반'] == target_class) & 
        (df['분리대상'] == my_name)
    ]
    if not haters.empty:
        return False # 위험!
        
    return True # 안전

def allocate_class_logic(df):
    """학반별 순환 S자 배정 (기본 배정)"""
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
    """
    [1단계] 분리 배정 강제 적용
    초기 배정 후, 금지된 만남(같은 반)이 있으면 즉시 떼어놓음
    """
    classes = ['가', '나', '다']
    
    # 분리 대상이 있는 학생들만 필터링
    conflict_rows = df[df['분리대상'].notna()]
    
    for idx, row in conflict_rows.iterrows():
        enemy_name = row['분리대상']
        my_class = row['신학년반']
        
        # 앙숙이 같은 반에 있는지 확인
        enemies = df[
            (df['이름'] == enemy_name) & 
            (df['신학년반'] == my_class)
        ]
        
        if not enemies.empty:
            # 같은 반에 앙숙이 있음! -> '나'를 다른 반으로 이동
            # 이동할 반 후보 찾기 (앙숙이 없는 반)
            available_classes = [c for c in classes if c != my_class]
            
            for target_class in available_classes:
                # 이동하려는 반에도 또 다른 앙숙이 없는지 체크
                if check_conflict_safety(df, idx, target_class):
                    # 안전하다면, target_class의 '일반 학생'과 맞교환 (인원수 유지)
                    # 성별 같고, 점수 비슷한 일반 학생 찾기
                    swap_candidates = df[
                        (df['신학년반'] == target_class) &
                        (df['생활지도_표시'] == False) &
                        (df['성별'] == row['성별'])
                    ]
                    
                    if not swap_candidates.empty:
                        # 점수 차이 최소인 학생
                        swap_candidates = swap_candidates.copy()
                        swap_candidates['diff'] = abs(swap_candidates['총점'] - row['총점'])
                        target_student = swap_candidates.sort_values('diff').iloc[0]
                        target_idx = target_student.name
                        
                        # 교환 실행
                        df.at[idx, '신학년반'] = target_class
                        df.at[target_idx, '신학년반'] = my_class
                        break # 해결 완료
    return df

def distribute_special_students_global(df):
    """
    [2단계] 생활지도 학생 수 균형 맞추기 (4~6명)
    단, 교환 시 '분리 배정 원칙'을 위반하지 않아야 함
    """
    max_iter = 300
    
    for i in range(max_iter):
        counts = df[df['생활지도_표시'] == True]['신학년반'].value_counts()
        for cls in ['가', '나', '다']:
            if cls not in counts: counts[cls] = 0
            
        max_val = counts.max()
        min_val = counts.min()
        
        if max_val - min_val <= 1:
            break
            
        src_class = counts.idxmax()
        dst_class = counts.idxmin()
        
        # 교환 후보 (과밀반의 생활지도 학생)
        src_candidates = df[
            (df['신학년반'] == src_class) & 
            (df['생활지도_표시'] == True)
        ]
        
        best_swap_pair = None
        min_score_diff = float('inf')
        
        for src_idx, src_student in src_candidates.iterrows():
            # [중요] 이 학생을 dst_class로 보내도 안전한가? (앙숙 체크)
            if not check_conflict_safety(df, src_idx, dst_class):
                continue
                
            s_gender = src_student['성별']
            s_score = src_student['총점']
            
            # 맞교환 대상 (부족반의 일반 학생)
            dst_candidates = df[
                (df['신학년반'] == dst_class) & 
                (df['생활지도_표시'] == False) & 
                (df['성별'] == s_gender)
            ]
            
            if dst_candidates.empty: continue
            
            # [중요] 맞교환 대상 학생을 src_class로 가져와도 안전한가?
            # (일반 학생이라도 누군가의 기피 대상일 수 있음)
            safe_targets = []
            for d_idx, d_row in dst_candidates.iterrows():
                 if check_conflict_safety(df, d_idx, src_class):
                     safe_targets.append(d_row)
            
            if not safe_targets: continue
            
            # 안전한 대상들 중에서 점수 차이 계산
            safe_df = pd.DataFrame(safe_targets)
            safe_df['diff'] = abs(safe_df['총점'] - s_score)
            best_match = safe_df.sort_values('diff').iloc[0]
            
            if best_match['diff'] < min_score_diff:
                min_score_diff = best_match['diff']
                best_swap_pair = (src_idx, best_match.name)
        
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
            
            # 2. [New] 앙숙 관계(분리 배정) 우선 해결
            df_allocated = resolve_conflicts_first(df_allocated)
            
            # 3. 생활지도 학생 수 균형 조절 (분리 원칙 준수 하에)
            df_final = distribute_special_students_global(df_allocated)
            
            # 4. 비고 생성 (분리 대상이 있으면 함께 표시)
            def make_note(row):
                notes = []
                if row['생활지도_표시']: notes.append('★생활지도')
                if row['분리대상']: notes.append(f"(분리:{row['분리대상']})")
                return ' '.join(notes)
                
            df_final['비고'] = df_final.apply(make_note, axis=1)
            
            st.session_state.df_result = df_final
            st.success("✅ 반편성 완료! (분리 배정 및 생활지도 균형 적용)")
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
        df_display['선택라벨'] = df_display.apply(lambda x: f"{x['이름']} ({x['신학년반']} / {x['총점']}점)", axis=1)
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
                        
                        # 수동 교환 시 경고 (안전 체크)
                        safe_a = check_conflict_safety(st.session_state.df_result, idx_a, row_b['신학년반'])
                        safe_b = check_conflict_safety(st.session_state.df_result, idx_b, row_a['신학년반'])
                        
                        if not (safe_a and safe_b):
                            st.warning("⚠️ 주의: 이 교환은 분리 배정 원칙(생활지도 곤란 학생 분리)과 충돌할 수 있습니다.")
                        
                        val_a = st.session_state.df_result.at[idx_a, '신학년반']
                        val_b = st.session_state.df_result.at[idx_b, '신학년반']
                        st.session_state.df_result.at[idx_a, '신학년반'] = val_b
                        st.session_state.df_result.at[idx_b, '신학년반'] = val_a
                        
                        st.success("교환 완료!"); st.rerun()
                    except Exception as e:
                        st.error(f"오류: {e}")
                else:
                    st.warning("다른 학생을 선택해주세요.")

    tabs = st.tabs(["가반", "나반", "다반", "전체"])
    
    def show_tab(cls_name):
        subset = df_display[df_display['신학년반'] == cls_name][cols]
        count = len(subset)
        special = len(subset[subset['비고'].str.contains('생활지도')])
        avg = subset['총점'].mean() if count > 0 else 0
        
        msg = f"👥 총원: {count}명 | ⚠️ 생활지도: {special}명 | 📊 평균점수: {avg:.1f}점"
        
        if 4 <= special <= 6: st.success(msg + " (적정)")
        else: st.warning(msg + " (조정 권장)")
        
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
