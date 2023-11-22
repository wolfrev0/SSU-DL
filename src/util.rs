pub fn is_equal<'a, 'b>(
	mut a: impl Iterator<Item = &'a f32>,
	mut b: impl Iterator<Item = &'b f32>,
) -> bool {
	while let Some(x) = a.next() {
		if let Some(y) = b.next() {
			if (x - y).abs() >= 0.001 {
				return false;
			} else {
				continue;
			}
		} else {
			return false;
		}
	}
	b.next() == None
}
