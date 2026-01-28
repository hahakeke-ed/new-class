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

st.title("🏫 2026학년도 초등학교 반편성 시스템 (최종버전)")
st.markdown("""
**반편성 핵심 원칙:**
1. **[0순위] 동명이인 분리:** 이름이 같은 학생은 무조건 다른 반 배정
2. **[1순위] 분리 요청 해결:** '분리요청' 칸에 적힌 학생과는 절대 같은 반 배정 금지
3. **[2순위] 구학년 안배:** 반별 구학년 동성 친구 최소 4명 이상 유지
4. **[3순위] 생활지도 균형:** '생활지도 곤란' 학생 반별 균등(4~6명) 배치
""")

# --------------------------------------------------------------------------
# 2. 데이터 처리 및 알고리즘 함수
# --------------------------------------------------------------------------
def preprocess_data(df):
    """
    데이터 정제: 생활지도와 분리요청을 별도 컬럼으로 처리
    """
    # 1. 컬럼명 매핑 (업로드한 파일 기준)
    # 파일의 헤더: 학반, 번호, 성별, 성명, 시험1, 시험2, 합, 생활지도 곤란, 분리요청
    col_map = {
        '학반': '2025반',
        '번호': '2025번호',
        '성명': '이름',
        '합': '총점',
        '생활지도 곤란': '생활지도',
        '분리요청': '분리요청'
    }
    # 매핑되지 않은 나머지 컬럼은 그대로 둠
    df = df.rename(columns=col_map)
    
    # 2. 필수 컬럼 체크
    required = ['2025반', '2025번호', '성별', '이름', '총점']
    if not all(col in df.columns for col in required):
        return None, f"필수 컬럼이 누락되었습니다. (필요: {required}, 현재: {list(df.columns)})"

    df = df.dropna(subset=['이름'])
    
    # 3. 점수 및 번호 정수 변환
    df['총점'] = pd.to_numeric(df['총점'], errors='coerce')
    avg_score = df['총점'].mean()
    if pd.isna(avg_score): avg_score = 0 
    df['총점'] = df['총점'].fillna(avg_score).round().astype(int)
    
    for col in ['2025반', '2025번호']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # 4. [핵심] 생활지도 vs 분리요청 이원화 처리
    
    # (1) 생활지도 여부 (Behavior): 값이 있으면 True (점수든 O든)
    if '생활지도' in df.columns:
        df['is_behavior'] = df['생활지도'].astype(str).apply(
            lambda x: True if x.strip() not in ['nan', '', 'None', '0', '0.0'] else False
        )
    else:
        df['is_behavior'] = False
        
    # (2) 분리 요청 대상 (Separation): 텍스트가 있으면 그 이름을 저장
    if '분리요청' in df.columns:
        def extract_name(val):
            s = str(val).strip()
            if s in ['nan', '', 'None', '0', '0.0']: return None
            # 숫자로만 된 게 아니면 이름으로 간주
            if not s.replace('.', '').isdigit(): return s
            return None
        df['conflict_target'] = df['분리요청'].apply(extract_name)
    else:
        df['conflict_target'] = None
        
    return df, None

# --- 제약 조건 검사 함수들 ---

def check_homonym_safety(df, student_idx, target_class):
    """동명이인 체크: target_class에 내 이름과 같은 학생이 이미 있는가?"""
    my_name = df.loc[student_idx, '이름']
    # 나 자신 제외하고 검색
    same_names = df[
        (df['신학년반'] == target_class) & 
        (df['이름'] == my_name) & 
        (df.index != student_idx)
    ]
    if not same_names.empty:
        return False # 위험
    return True

def check_conflict_safety(df, student_idx, target_class):
    """분리 요청 체크 (쌍방향 확인)"""
    student = df.loc[student_idx]
    
    # 1. 내가 피하고 싶은 사람이 저기에 있는가?
    enemy_name = student['conflict_target']
    if enemy_name:
        enemies = df[(df['이름'] == enemy_name) & (df['신학년반'] == target_class)]
        if not enemies.empty: return False # 위험

    # 2. 저기에 있는 누군가가 나를 피하고 싶은가?
    my_name = student['이름']
    haters = df[(df['신학년반'] == target_class) & (df['conflict_target'] == my_name)]
    if not haters.empty: return False # 위험
        
    return True

def check_old_class_constraint(df, student_idx, current_class, min_count=4):
    """구학년 동성 친구 최소 인원(4명) 보존 체크"""
    student = df.loc[student_idx]
    old_cls = student['2025반']
    gender = student['성별']
    
    # 현재 반에 남아있는 내 친구들 수 (나 포함)
    count = len(df[
        (df['신학년반'] == current_class) & 
        (df['2025반'] == old_cls) &
        (df['성별'] == gender)
    ])
    
    # 내가 나가면 (count-1)명이 됨. 
    # 즉, 현재 count가 min_count보다 커야만 나갈 수 있음.
    # (이미 4명 이하라면 더 줄일 수 없음)
    if count <= min_count:
        return False
    return True

# --- 배정 로직 ---

def allocate_initial(df):
    """1단계: 학반별 순환 S자 배정 (기본)"""
    results = []
    # 2025반, 성별로 그룹핑하여 성적순 정렬
    for (old_cls, gender), sub in df.groupby(['2025반', '성별']):
        sub = sub.sort_values(by=['총점', '이름'], ascending=[False, True]).copy()
        
        # 로테이션 규칙
        if old_cls == 1: targets = ['가', '나', '다']
        elif old_cls == 2: targets = ['나', '다', '가']
        elif old_cls == 3: targets = ['다', '가', '나']
        else: targets = ['가', '나', '다']
            
        new_classes = []
        for i in range(len(sub)):
            # S자 패턴: 0,1,2, 2,1,0
            cycle = [0, 1, 2, 2, 1, 0][i % 6]
            new_classes.append(targets[cycle])
        
        sub['신학년반'] = new_classes
        results.append(sub)
        
    if not results: return df
    return pd.concat(results, ignore_index=True)

def solve_homonyms(df):
    """2단계: 동명이인 강제 분리"""
    names = df['이름'].value_counts()
    homonyms = names[names > 1].index.tolist()
    classes = ['가', '나', '다']
    
    for name in homonyms:
        students = df[df['이름'] == name]
        cls_counts = students['신학년반'].value_counts()
        
        for cls, cnt in cls_counts.items():
            if cnt > 1:
                # 한 반에 2명 이상 -> 이동 필요
                targets = students[students['신학년반'] == cls]
                # 첫명 빼고 나머지 이동
                movers = targets.iloc[1:]
                
                for idx, row in movers.iterrows():
                    current = row['신학년반']
                    # 이동 가능한 반 (동명이인 없는 곳)
                    candidates = [c for c in classes if c != current and check_homonym_safety(df, idx, c)]
                    
                    swapped = False
                    for target_cls in candidates:
                        # 이동 시 안전 체크
                        if not check_conflict_safety(df, idx, target_cls): continue
                        if not check_old_class_constraint(df, idx, current): continue
                        
                        # 스왑 파트너 찾기 (성별 같고, 일반 학생 우선)
                        swap_pool = df[
                            (df['신학년반'] == target_cls) &
                            (df['성별'] == row['성별']) &
                            (df['이름'] != name) # 내 이름 아닌 사람
                        ].copy()
                        
                        # 점수차 정렬
                        swap_pool['diff'] = abs(swap_pool['총점'] - row['총점'])
                        swap_pool = swap_pool.sort_values('diff')
                        
                        for s_idx, s_row in swap_pool.iterrows():
                            # 파트너 안전 체크
                            if not check_conflict_safety(df, s_idx, current): continue
                            if not check_old_class_constraint(df, s_idx, target_cls): continue
                            if not check_homonym_safety(df, s_idx, current): continue
                            
                            # 교환
                            df.at[idx, '신학년반'] = target_cls
                            df.at[s_idx, '신학년반'] = current
                            swapped = True
                            break
                        if swapped: break
    return df

def solve_separations(df):
    """3단계: 분리 요청(앙숙) 해결"""
    classes = ['가', '나', '다']
    # 분리 요청이 있는 학생만 필터
    conflicts = df[df['conflict_target'].notna()]
    
    for idx, row in conflicts.iterrows():
        enemy = row['conflict_target']
        current = row['신학년반']
        
        # 같은 반에 앙숙이 있는지 확인
        enemies_in_class = df[
            (df['이름'] == enemy) & (df['신학년반'] == current)
        ]
        
        if not enemies_in_class.empty:
            # 이동 필요
            others = [c for c in classes if c != current]
            
            for target_cls in others:
                if not check_conflict_safety(df, idx, target_cls): continue
                if not check_old_class_constraint(df, idx, current): continue
                if not check_homonym_safety(df, idx, target_cls): continue
                
                # 스왑 파트너
                swap_pool = df[
                    (df['신학년반'] == target_cls) &
                    (df['성별'] == row['성별'])
                ].copy()
                swap_pool['diff'] = abs(swap_pool['총점'] - row['총점'])
                swap_pool = swap_pool.sort_values('diff')
                
                for s_idx, s_row in swap_pool.iterrows():
                    if not check_conflict_safety(df, s_idx, current): continue
                    if not check_old_class_constraint(df, s_idx, target_cls): continue
                    if not check_homonym_safety(df, s_idx, current): continue
                    
                    df.at[idx, '신학년반'] = target_cls
                    df.at[s_idx, '신학년반'] = current
                    break # 다음 앙숙 해결로
    return df

def balance_behavior(df):
    """4단계: 생활지도 곤란 학생 균형 (4~6명)"""
    max_iter = 300
    
    for _ in range(max_iter):
        counts = df[df['is_behavior'] == True]['신학년반'].value_counts()
        for c in ['가', '나', '다']: 
            if c not in counts: counts[c] = 0
            
        if counts.max() - counts.min() <= 1: break # 균형 도달
        
        src_cls = counts.idxmax()
        dst_cls = counts.idxmin()
        
        # 과밀반의 생활지도 학생들
        candidates = df[
            (df['신학년반'] == src_cls) & 
            (df['is_behavior'] == True)
        ]
        
        best_pair = None
        min_diff = float('inf')
        
        for idx, row in candidates.iterrows():
            # 이동 안전 체크
            if not check_conflict_safety(df, idx, dst_cls): continue
            if not check_old_class_constraint(df, idx, src_cls): continue
            if not check_homonym_safety(df, idx, dst_cls): continue
            
            # 부족반의 일반 학생 찾기
            targets = df[
                (df['신학년반'] == dst_cls) &
                (df['is_behavior'] == False) &
                (df['성별'] == row['성별'])
            ]
            
            for t_idx, t_row in targets.iterrows():
                if not check_conflict_safety(df, t_idx, src_cls): continue
                if not check_old_class_constraint(df, t_idx, dst_cls): continue
                if not check_homonym_safety(df, t_idx, src_cls): continue
                
                diff = abs(t_row['총점'] - row['총점'])
                if diff < min_diff:
                    min_diff = diff
                    best_pair = (idx, t_idx)
        
        if best_pair:
            df.at[best_pair[0], '신학년반'] = dst_cls
            df.at[best_pair[1], '신학년반'] = src_cls
        else:
            break # 교환 불가 시 중단
            
    return df

# --------------------------------------------------------------------------
# 3. 메인 앱 UI 및 로직 실행
# --------------------------------------------------------------------------
if 'df_result' not in st.session_state:
    st.session_state.df_result = None

uploaded_file = st.file_uploader("최신 양식의 엑셀 파일을 업로드하세요 (생활지도/분리요청 컬럼 포함)", type=['xlsx', 'csv'])

if uploaded_file is not None and st.session_state.df_result is None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
            
        df, error_msg = preprocess_data(df_raw)
        
        if error_msg:
            st.error(error_msg)
        else:
            # 알고리즘 실행 파이프라인
            with st.spinner('반편성 알고리즘 수행 중...'):
                # 1. 초기 배정
                df = allocate_initial(df)
                df = df.reset_index(drop=True)
                
                # 2. 동명이인 분리
                df = solve_homonyms(df)
                
                # 3. 분리 요청 해결
                df = solve_separations(df)
                
                # 4. 생활지도 균형
                df = balance_behavior(df)
                
                # 비고 작성
                def make_note(row):
                    notes = []
                    # 동명이인 확인
                    if len(df[df['이름'] == row['이름']]) > 1:
                        notes.append("★동명이인")
                    if row['is_behavior']:
                        notes.append("★생활지도")
                    if row['conflict_target']:
                        notes.append(f"(분리:{row['conflict_target']})")
                    return " ".join(notes)
                
                df['비고'] = df.apply(make_note, axis=1)
                
                st.session_state.df_result = df
                st.success("✅ 반편성 완료! (생활지도/분리요청 이원화 적용)")
                st.rerun()
                
    except Exception as e:
        st.error(f"오류 발생: {e}")

# 결과 화면
if st.session_state.df_result is not None:
    df_disp = st.session_state.df_result.copy()
    
    # 정렬
    df_disp['성별_order'] = df_disp['성별'].apply(lambda x: 0 if x != '남' else 1)
    df_disp = df_disp.sort_values(by=['신학년반', '성별_order', '이름']).reset_index(drop=True)
    
    cols = ['신학년반', '이름', '성별', '2025반', '2025번호', '총점', '비고']
    
    # 엑셀 다운로드
    col_h, col_b = st.columns([3, 1])
    with col_h: st.subheader("📋 최종 결과물")
    with col_b:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_disp[cols].to_excel(writer, index=False, sheet_name='반편성결과')
        st.download_button("📥 엑셀 다운로드", data=output.getvalue(), file_name="2026_반편성_완료.xlsx", type="primary")

    st.divider()

    # 수동 교환 UI
    with st.expander("🛠️ 수동 교환 (관리자용)", expanded=True):
        df_disp['label'] = df_disp.apply(lambda x: f"{x['이름']} ({x['신학년반']} / {x['총점']}점)", axis=1)
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1: s_a = st.selectbox("학생 A", df_disp['label'].unique(), key='a')
        with c2: s_b = st.selectbox("학생 B", df_disp['label'].unique(), key='b')
        with c3:
            st.write(""); st.write("")
            if st.button("교환 실행"):
                if s_a != s_b:
                    # 인덱스 찾기
                    r_a = df_disp[df_disp['label'] == s_a].iloc[0]
                    r_b = df_disp[df_disp['label'] == s_b].iloc[0]
                    
                    # 고유 식별 (이름+구반+번호)
                    idx_a = st.session_state.df_result[
                        (st.session_state.df_result['이름'] == r_a['이름']) &
                        (st.session_state.df_result['2025반'] == r_a['2025반']) &
                        (st.session_state.df_result['2025번호'] == r_a['2025번호'])
                    ].index[0]
                    idx_b = st.session_state.df_result[
                        (st.session_state.df_result['이름'] == r_b['이름']) &
                        (st.session_state.df_result['2025반'] == r_b['2025반']) &
                        (st.session_state.df_result['2025번호'] == r_b['2025번호'])
                    ].index[0]
                    
                    # 안전 체크 (경고만)
                    warns = []
                    # 동명이인
                    if not check_homonym_safety(st.session_state.df_result, idx_a, r_b['신학년반']): warns.append("A가 가면 동명이인 발생")
                    if not check_homonym_safety(st.session_state.df_result, idx_b, r_a['신학년반']): warns.append("B가 오면 동명이인 발생")
                    # 분리요청
                    if not check_conflict_safety(st.session_state.df_result, idx_a, r_b['신학년반']): warns.append("A의 분리요청 위반")
                    if not check_conflict_safety(st.session_state.df_result, idx_b, r_a['신학년반']): warns.append("B의 분리요청 위반")
                    # 구반인원
                    if not check_old_class_constraint(st.session_state.df_result, idx_a, r_a['신학년반']): warns.append("A반 구학년 인원 부족")
                    if not check_old_class_constraint(st.session_state.df_result, idx_b, r_b['신학년반']): warns.append("B반 구학년 인원 부족")
                    
                    if warns:
                        st.warning("⚠️ 주의: " + ", ".join(warns))
                    
                    # 교환
                    va = st.session_state.df_result.at[idx_a, '신학년반']
                    vb = st.session_state.df_result.at[idx_b, '신학년반']
                    st.session_state.df_result.at[idx_a, '신학년반'] = vb
                    st.session_state.df_result.at[idx_b, '신학년반'] = va
                    st.success("교환되었습니다.")
                    st.rerun()

    # 탭별 보기
    tabs = st.tabs(["가반", "나반", "다반", "전체"])
    
    def show_stats(cls_name):
        sub = df_disp[df_disp['신학년반'] == cls_name]
        cnt = len(sub)
        # 생활지도 카운트 (is_behavior 컬럼 기준)
        # 원본 데이터프레임에서 가져와야 정확함 (비고는 텍스트라)
        real_sub = st.session_state.df_result[st.session_state.df_result['신학년반'] == cls_name]
        beh_cnt = len(real_sub[real_sub['is_behavior'] == True])
        
        avg = sub['총점'].mean() if cnt > 0 else 0
        
        # 구반 분포
        dist = sub.groupby(['2025반', '성별']).size().unstack(fill_value=0)
        dist_str = []
        err = False
        for c_idx, row in dist.iterrows():
            m = row.get('남', 0)
            f = row.get('여', 0)
            dist_str.append(f"{c_idx}반({m}/{f})")
            if (m > 0 and m < 4) or (f > 0 and f < 4): err = True
            
        st.info(f"👥 {cnt}명 | 생활지도: {beh_cnt}명 | 평균: {avg:.1f}")
        if err: st.error(f"구학년 분포: {', '.join(dist_str)} (4명 미만 주의)")
        else: st.success(f"구학년 분포: {', '.join(dist_str)}")
        
        st.dataframe(
            sub[cols].style.apply(lambda x: ['background-color: #ffcccc' if '★' in str(v) else '' for v in x], subset=['비고'], axis=1),
            use_container_width=True, hide_index=True, height=600
        )

    with tabs[0]: show_stats('가')
    with tabs[1]: show_stats('나')
    with tabs[2]: show_stats('다')
    with tabs[3]: st.dataframe(df_disp[cols], use_container_width=True, height=800)
    
    if st.button("초기화"):
        st.session_state.df_result = None
        st.rerun()
