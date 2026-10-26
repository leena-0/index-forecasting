# 협업 가이드

## 작업 흐름

1. GitHub Issue에서 `기능(feat)`, `버그(fix)`, `작업(chore)` 중 알맞은 양식을 선택한다.
2. 최신 `main`에서 이슈 단위 브랜치를 만든다. 별도의 `develop` 브랜치는 사용하지 않는다.
3. 변경과 테스트를 커밋하고 Pull Request를 연다.
4. CI 통과와 리뷰 승인을 확인한 뒤 병합한다.

브랜치 이름은 `<유형>/<이슈번호>-<짧은-설명>` 형식을 권장한다.

```text
feat/12-add-event-loader
fix/24-handle-missing-price
chore/31-update-docs
```

커밋 제목은 `<유형>: <요약>` 형식을 사용한다. 허용하는 유형은 다음과 같다.

- `feat`: 새로운 기능
- `fix`: 버그 수정
- `refactor`: 동작 변경 없는 코드 구조 개선
- `docs`: 문서 변경
- `test`: 테스트 추가 및 수정
- `chore`: 설정, 의존성 등 기타 유지보수

```text
feat: add event announcement loader
fix: handle missing trading days
refactor: simplify window assignment
docs: document data collection process
test: add missing-price test cases
chore: update setup documentation
```

## Pull Request 규칙

- 한 PR에는 하나의 목적만 담는다.
- PR 대상 브랜치는 `main`으로 지정한다.
- 본문에 `Closes #이슈번호`를 적어 관련 이슈를 연결한다.
- 제출 전에 `python -m unittest`를 실행한다.
- 동작이나 사용법이 바뀌면 관련 문서도 함께 수정한다.
- API 키, `.env`, 개인 데이터 및 `data/` 산출물을 커밋하지 않는다.
- 연구 설계의 `[확정]` 사항을 바꾸는 PR은 변경 이유와 영향을 먼저 명시한다.

## 리뷰 기준

- 미래 정보 누출 방지 등 `CLAUDE.md`의 확정 규칙을 지키는가?
- 합성 데이터 기반 테스트가 추가되거나 기존 테스트가 유지되는가?
- 실제 네트워크와 개인 API 키 없이 CI가 재현 가능한가?
- 변경 범위와 이름이 명확하고 문서와 코드가 일치하는가?
