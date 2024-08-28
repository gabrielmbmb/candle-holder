use serde::{Deserialize, Deserializer};
use std::fmt::Debug;
use std::str::FromStr;

pub fn deserialize_single_or_vec<'de, D, T>(deserializer: D) -> Result<Option<Vec<T>>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de> + FromStr + Debug,
    <T as FromStr>::Err: Debug,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum SingleOrVec<T> {
        Single(T),
        Vec(Vec<T>),
    }

    match Option::<SingleOrVec<T>>::deserialize(deserializer)? {
        Some(SingleOrVec::Vec(vec)) => Ok(Some(vec)),
        Some(SingleOrVec::Single(single)) => Ok(Some(vec![single])),
        None => Ok(None),
    }
}
