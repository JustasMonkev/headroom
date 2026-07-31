"""Tests for compression summary generation."""

from headroom.transforms.compression_summary import (
    _extract_name_from_signature,
    summarize_compressed_code,
)


class TestSummarizeCompressedCode:
    def test_python_function_bodies(self):
        bodies = [
            ("def authenticate(username, password):", "    db = get_db()\n    return True", 10),
            ("def validate_token(token):", "    return jwt.decode(token)", 20),
            ("def refresh_session(user):", "    session.extend()", 30),
        ]
        summary = summarize_compressed_code(bodies, 3)
        assert summary.startswith("3 bodies: ")
        assert "authenticate" in summary
        assert "validate_token" in summary

    def test_javascript_function_bodies(self):
        bodies = [
            ("function handleRequest(req, res) {", "  res.send('ok');", 5),
            ("async function fetchData(url) {", "  return await fetch(url);", 15),
        ]
        summary = summarize_compressed_code(bodies, 2)
        assert "handleRequest" in summary
        assert "fetchData" in summary

    def test_go_function_bodies(self):
        bodies = [
            (
                "func (s *Server) HandleRequest(w http.ResponseWriter, r *http.Request) {",
                '  w.Write([]byte("ok"))',
                10,
            ),
            ("func main() {", "  server.Start()", 1),
        ]
        summary = summarize_compressed_code(bodies, 2)
        assert "HandleRequest" in summary
        assert "main" in summary

    def test_rust_function_bodies(self):
        bodies = [
            ("fn authenticate(token: &str) -> Result<User, Error> {", "  Ok(User::new())", 10),
        ]
        summary = summarize_compressed_code(bodies, 1)
        assert "authenticate" in summary

    def test_empty_bodies(self):
        summary = summarize_compressed_code([], 0)
        assert summary == ""

    def test_many_bodies_truncated(self):
        """The name list is capped at 6 with NO `(+N more)` suffix.

        The suffix cost ~5 tokens to announce that a list of examples is a
        list of examples; the leading count already says how many there were.
        """
        bodies = [(f"def func_{i}(x):", f"    return {i}", i * 10) for i in range(20)]
        summary = summarize_compressed_code(bodies, 20)
        assert summary == "20 bodies: func_0, func_1, func_2, func_3, func_4, func_5"
        assert "more" not in summary


class TestExtractNameFromSignature:
    def test_python_def(self):
        assert _extract_name_from_signature("def authenticate(username):") == "authenticate"

    def test_python_async_def(self):
        assert _extract_name_from_signature("async def fetch_data(url):") == "fetch_data"

    def test_javascript_function(self):
        assert _extract_name_from_signature("function handleClick(event) {") == "handleClick"

    def test_go_func(self):
        assert (
            _extract_name_from_signature("func HandleRequest(w http.ResponseWriter) {")
            == "HandleRequest"
        )

    def test_go_method(self):
        assert _extract_name_from_signature("func (s *Server) Start() {") == "Start"

    def test_rust_fn(self):
        assert (
            _extract_name_from_signature("fn authenticate(token: &str) -> Result<User> {")
            == "authenticate"
        )

    def test_java_method(self):
        assert (
            _extract_name_from_signature("public void processPayment(Payment p) {")
            == "processPayment"
        )

    def test_class(self):
        assert _extract_name_from_signature("class TokenValidator:") == "TokenValidator"

    def test_empty(self):
        assert _extract_name_from_signature("") == ""

    def test_export_async(self):
        assert _extract_name_from_signature("export async function fetchUsers() {") == "fetchUsers"
