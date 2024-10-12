
import { BusinessExplore } from "@/app/modules/business/BusinessExplore";
import { AppBar, Toolbar } from "@mui/material";


export default function forYouPage() {
    const MOCK_DATA = [
        {
            id: 1,
            name: "McDonald's",
            description: "Fast food chain"
        },
        {
            id: 2,
            name: "Burger King",
            description: "Fast food chain"
        },
        {
            id: 3,
            name: "Wendy's",
            description: "Fast food chain"
        }
    ];

    return (
        <div>
           <BusinessExplore businesses={MOCK_DATA} />
        </div>
    );
}