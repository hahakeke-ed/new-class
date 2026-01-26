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
        # 여기서는 단순 변환 시도
        if not old_class.isdigit():
             # "1"이 포함되어 있으면 1로 간주하는 식의 단순 처리 (데이터 오염 대비)
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
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
            
        # 전처리
        df, error_msg = preprocess_data(df_raw)
        
        if error_msg:
            st.error(error_msg)
        else:
            st.success(f"{len(df)}명의 데이터를 불러왔습니다. 반편성을 시작합니다.")
            
            # 1. 남녀 분리
            df_male = df[df['성별'] == '남'].copy()
            df_female = df[df['성별'] != '남'].copy() # '여' 또는 기타
            
            # 2. 그룹핑 (성적순 S자)
            df_male = s_shape_grouping(df_male)
            df_female = s_shape_grouping(df_female)
            
            # 3. 합치기
            result_df = pd.concat([df_male, df_female])
            
            # 4. 반 배정
            result_df['신학년반'] = result_df.apply(assign_new_class, axis=1)
            
            # 5. 최종 정렬 (반 > 성별(여학생우선) > 이름)
            # 여학생 우선 정렬을 위해 플래그 생성
            result_df['성별_order'] = result_df['성별'].apply(lambda x: 0 if x != '남' else 1)
            result_df = result_df.sort_values(by=['신학년반', '성별_order', '이름'])
            
            # 6. 화면 표시용 데이터 정리
            # 생활지도 곤란 학생은 비고란에 ★ 표시
            result_df['비고'] = result_df['생활지도_표시'].apply(lambda x: '★생활지도(3점)' if x else '')
            
            display_cols = ['신학년반', '이름', '성별', '현재반', '총점', '그룹', '비고']
            
            # ------------------------------------------------------------------
            # 결과 탭 보기
            # ------------------------------------------------------------------
            st.divider()
            st.subheader("📋 반편성 결과 (미리보기)")
            
            tabs = st.tabs(["가반", "나반", "다반", "전체 명부"])
            
            def show_class_table(class_name):
                subset = result_df[result_df['신학년반'] == class_name][display_cols]
                
                # 생활지도 대상자가 몇 명인지 카운트
                target_count = len(subset[subset['비고'] != ''])
                if target_count > 0:
                    st.warning(f"⚠️ 이 반에는 생활지도 고려 학생이 {target_count}명 포함되어 있습니다.")
                
                # 데이터프레임 표시 (특정 행 강조는 스트림잇 기본 기능 한계로 비고 컬럼 활용)
                st.dataframe(
                    subset.style.apply(lambda x: ['background-color: #ffcccc' if v != '' else '' for v in x], subset=['비고'], axis=1),
                    use_container_width=True,
                    hide_index=True
                )
            
            with tabs[0]: show_class_table('가')
            with tabs[1]: show_class_table('나')
            with tabs[2]: show_class_table('다')
            with tabs[3]: 
                st.dataframe(result_df[display_cols], use_container_width=True)

            # ------------------------------------------------------------------
            # 엑셀 다운로드
            # ------------------------------------------------------------------
            st.divider()
            
            # 다운로드용 파일 생성
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                result_df[display_cols].to_excel(writer, index=False, sheet_name='반편성결과')
                
            st.download_button(
                label="📥 최종 결과 엑셀 다운로드",
                data=output.getvalue(),
                file_name="2026_반편성_완료.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"처리 중 오류가 발생했습니다: {e}")
