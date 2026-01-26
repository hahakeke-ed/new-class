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
    </style>
""", unsafe_allow_html=True)

st.title("🏫 2026학년도 초등학교 반편성 시스템")
st.markdown("첨부된 **반편성 계획**에 의거하여 성적순 S자 배치 및 반 배정을 자동화합니다.")

# --------------------------------------------------------------------------
# 2. 데이터 처리 함수
# --------------------------------------------------------------------------
def preprocess_data(df):
    """
    업로드된 엑셀/CSV 데이터를 표준 포맷으로 변환
    """
    # 1. 컬럼명 매핑 (사용자 파일 -> 앱 내부 표준 변수명)
    col_map = {
        '성명': '이름',
        '합': '총점',
        '학반': '현재반',
        '생활지도 곤란': '생활지도' # 3.0 등으로 입력된 컬럼
    }
    df = df.rename(columns=col_map)
    
    # 2. 필수 컬럼 존재 여부 체크
    required = ['이름', '성별', '총점', '현재반']
    if not all(col in df.columns for col in required):
        return None, f"필수 컬럼이 누락되었습니다. (필요: {required}, 현재: {list(df.columns)})"

    # 3. 데이터 정제
    # 결측치 처리 (이름이 없는 행 삭제)
    df = df.dropna(subset=['이름'])
    
    # 총점: 숫자로 변환, NaN은 0점으로
    df['총점'] = pd.to_numeric(df['총점'], errors='coerce').fillna(0)
    
    # 생활지도: NaN이 아니면(값이 있으면) True, 아니면 False
    # (파일에 3.0 등 점수가 있으면 '관리필요'로 인식)
    if '생활지도' in df.columns:
        df['생활지도_표시'] = df['생활지도'].notna() & (df['생활지도'] != 0)
    else:
        df['생활지도_표시'] = False
        
    return df, None

def s_shape_grouping(df):
    """성적순 S자(ㄹ자) 그룹핑"""
    # 성적순 정렬 (동점자 처리: 생년월일 등이 없으므로 이름순 보조 정렬)
    df = df.sort_values(by=['총점', '이름'], ascending=[False, True]).reset_index(drop=True)
    groups = []
    
    for i in range(len(df)):
        cycle = i % 6 
        # S자 패턴: A->B->C->C->B->A
        if cycle == 0: group = 'A'
        elif cycle == 1: group = 'B'
        elif cycle == 2: group = 'C'
        elif cycle == 3: group = 'C'
        elif cycle == 4: group = 'B'
        else: group = 'A'
        groups.append(group)
        
    df['그룹'] = groups
    
    # 등수(임시) 추가 - 확인용
    df['석차'] = df.index + 1
    return df

def assign_new_class(row):
    """구학년 반 -> 신학년 반 매핑 규칙"""
    # 현재반에서 숫자만 추출 (예: "1반" -> "1")
    try:
        old_class = str(row['현재반']).replace('반', '').strip()
        # 숫자 외의 문자가 섞여있을 경우를 대비해 첫 글자만 따오거나 정제 로직 필요할 수 있음
        if not old_class.isdigit():
             if '1' in old_class: old_class = '1'
             elif '2' in old_class: old_class = '2'
             elif '3' in old_class: old_class = '3'
    except:
        return "미배정"

    group = row['그룹']
    
    if old_class == '1':
        return {'A': '가', 'B': '다', 'C': '나'}.get(group, '미배정')
    elif old_class == '2':
        return {'A': '나', 'B': '가', 'C': '다'}.get(group, '미배정')
    elif old_class == '3':
        return {'A': '다', 'B': '나', 'C': '가'}.get(group, '미배정')
    return "미배정"

# --------------------------------------------------------------------------
# 3. 메인 앱 로직
# --------------------------------------------------------------------------
uploaded_file = st.file_uploader("학생 성적 엑셀 파일을 업로드하세요 (.xlsx)", type=['xlsx', 'csv'])

if uploaded_file is not None:
    try:
        # 파일 읽기 (xlsx, csv 모두 지원)
        if uploaded_file.name.
