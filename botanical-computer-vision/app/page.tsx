import Dashboard from "./Dashboard";
import data from "./dashboard-data.json";
import strictComparison from "./strict-comparison-data.json";

export default function Home() {
  return <Dashboard data={data} comparison={strictComparison} />;
}
